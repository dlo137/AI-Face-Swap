import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import { useState } from 'react';
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { VideoView, useVideoPlayer } from 'expo-video';

const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!;
const PIAPI_KEY = process.env.EXPO_PUBLIC_PIAPI_KEY!;
const CLOUDINARY_CLOUD_NAME = process.env.EXPO_PUBLIC_CLOUDINARY_CLOUD_NAME!;
const CLOUDINARY_UPLOAD_PRESET = process.env.EXPO_PUBLIC_CLOUDINARY_UPLOAD_PRESET!;
const BUCKET_NAME = 'faceswap-uploads';
const MAX_VIDEO_BYTES = 10 * 1024 * 1024; // 10 MB

async function uploadToSupabase(
  fileUri: string,
  fileName: string,
  mimeType: string,
): Promise<string> {
  console.log(`[upload] fetching local file: ${fileUri}`);
  const response = await fetch(fileUri);
  const blob = await response.blob();
  console.log(`[upload] blob size: ${blob.size} bytes, type: ${blob.type}`);

  const uploadUrl = `${SUPABASE_URL}/storage/v1/object/${BUCKET_NAME}/${fileName}`;
  console.log(`[upload] POSTing to: ${uploadUrl}`);

  const res = await fetch(uploadUrl, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      'Content-Type': mimeType,
    },
    body: blob,
  });

  const resText = await res.text();
  console.log(`[upload] response status: ${res.status}, body: ${resText}`);

  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status} — ${resText}`);
  }

  const publicUrl = `${SUPABASE_URL}/storage/v1/object/public/${BUCKET_NAME}/${fileName}`;
  console.log(`[upload] public URL: ${publicUrl}`);
  return publicUrl;
}

async function resizeImageIfNeeded(uri: string): Promise<string> {
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: 1024 } }],
    { compress: 0.85, format: ImageManipulator.SaveFormat.JPEG },
  );
  return result.uri;
}

async function waitForCloudinaryVideo(url: string, maxAttempts = 15): Promise<void> {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const res = await fetch(url, { method: 'HEAD' });
      console.log(`[cloudinary] transcode poll ${i + 1}: ${res.status}`);
      if (res.ok) return;
    } catch {}
    await new Promise<void>((resolve) => setTimeout(resolve, 3000));
  }
  throw new Error('Cloudinary video took too long to process');
}

async function uploadVideoToCloudinary(fileUri: string): Promise<string> {
  console.log('[cloudinary] cloud:', CLOUDINARY_CLOUD_NAME, 'preset:', CLOUDINARY_UPLOAD_PRESET);
  const formData = new FormData();
  // React Native blobs don't serialize in FormData — use the native file object format instead
  formData.append('file', { uri: fileUri, type: 'video/mp4', name: 'upload.mp4' } as any);
  formData.append('upload_preset', CLOUDINARY_UPLOAD_PRESET);
  formData.append('resource_type', 'video');

  const uploadUrl = `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/video/upload`;
  console.log('[cloudinary] POSTing to:', uploadUrl);

  const res = await fetch(uploadUrl, { method: 'POST', body: formData });

  const resText = await res.text();
  console.log(`[cloudinary] response status: ${res.status}, body: ${resText}`);

  if (!res.ok) throw new Error(`Cloudinary upload failed: ${res.status} — ${resText}`);

  const data = JSON.parse(resText);
  const publicId: string = data.public_id;
  console.log('[cloudinary] public_id:', publicId);

  return `https://res.cloudinary.com/${CLOUDINARY_CLOUD_NAME}/video/upload/f_mp4,vc_h264,w_720,h_1280,c_limit/${publicId}.mp4`;
}

function uniqueName(ext: string): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}.${ext}`;
}

async function submitFaceSwap(videoUrl: string, imageUrl: string): Promise<string> {
  const body = {
    model: 'Qubico/video-toolkit',
    task_type: 'face-swap',
    input: {
      swap_image: imageUrl,
      target_video: videoUrl,
      swap_faces_index: '0',
      target_faces_index: '0',
    },
  };
  console.log('[piapi] submitting task, body:', JSON.stringify(body));
  console.log('[piapi] key present:', Boolean(PIAPI_KEY), 'key prefix:', PIAPI_KEY?.slice(0, 8));

  const res = await fetch('https://api.piapi.ai/api/v1/task', {
    method: 'POST',
    headers: {
      'x-api-key': PIAPI_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  const resText = await res.text();
  console.log(`[piapi] submit response status: ${res.status}, body: ${resText}`);

  let data: any;
  try {
    data = JSON.parse(resText);
  } catch {
    throw new Error(`PiAPI returned non-JSON: ${resText}`);
  }

  const taskId: string | undefined = data?.data?.task_id;
  if (!taskId) {
    throw new Error(`No task_id in response: ${resText}`);
  }
  console.log('[piapi] task_id:', taskId);
  return taskId;
}

async function pollForResult(
  taskId: string,
  onAttempt: (n: number) => void,
): Promise<string> {
  const MAX_ATTEMPTS = 60;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    onAttempt(attempt);
    const res = await fetch(`https://api.piapi.ai/api/v1/task/${taskId}`, {
      headers: { 'x-api-key': PIAPI_KEY },
    });
    const resText = await res.text();
    console.log(`[poll] attempt ${attempt}, status: ${res.status}, body: ${resText}`);

    let data: any;
    try {
      data = JSON.parse(resText);
    } catch {
      throw new Error(`Poll returned non-JSON: ${resText}`);
    }

    const status: string = data?.data?.status;
    console.log(`[poll] task status: ${status}`);

    if (status === 'completed') {
      const videoUrl: string | undefined = data?.data?.output?.video_url;
      if (!videoUrl) throw new Error(`No video_url in completed response: ${resText}`);
      console.log('[poll] result video URL:', videoUrl);
      return videoUrl;
    }
    if (status === 'failed') {
      throw new Error(`Face swap failed. Full response: ${resText}`);
    }

    if (attempt < MAX_ATTEMPTS) {
      await new Promise<void>((resolve) => setTimeout(resolve, 5000));
    }
  }
  throw new Error('Timed out waiting for face swap result');
}

export default function HomeScreen() {
  const [targetVideo, setTargetVideo] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [faceImage, setFaceImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [resultVideoUrl, setResultVideoUrl] = useState<string | null>(null);
  const [pollAttempt, setPollAttempt] = useState(0);

  const player = useVideoPlayer(resultVideoUrl, (p) => {
    p.loop = false;
  });

  async function pickVideo() {
    setError(null);
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      setError('Media library permission is required.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'videos',
      allowsEditing: false,
      quality: 1,
    });
    if (result.canceled || result.assets.length === 0) return;
    const asset = result.assets[0]!;
    if (asset.fileSize !== undefined && asset.fileSize > MAX_VIDEO_BYTES) {
      setTargetVideo(null);
      setError('Video must be under 10MB');
      return;
    }
    setTargetVideo(asset);
  }

  async function pickFaceImage() {
    setError(null);
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      setError('Media library permission is required.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images',
      allowsEditing: false,
      quality: 1,
    });
    if (result.canceled || result.assets.length === 0) return;
    setFaceImage(result.assets[0] ?? null);
  }

  async function handleSwap() {
    if (!targetVideo || !faceImage) return;
    setError(null);
    setResultVideoUrl(null);
    setPollAttempt(0);
    setIsLoading(true);
    try {
      setStatusMessage('Processing video...');
      const videoUrl = await uploadVideoToCloudinary(targetVideo.uri);

      setStatusMessage('Preparing video...');
      await waitForCloudinaryVideo(videoUrl);

      setStatusMessage('Resizing image...');
      const resizedUri = await resizeImageIfNeeded(faceImage.uri);

      setStatusMessage('Uploading image...');
      const imageUrl = await uploadToSupabase(
        resizedUri,
        uniqueName('jpg'),
        'image/jpeg',
      );

      setStatusMessage('Submitting to PiAPI...');
      const taskId = await submitFaceSwap(videoUrl, imageUrl);

      setStatusMessage('Processing face swap... (1/60)');
      const resultUrl = await pollForResult(taskId, (n) => {
        setPollAttempt(n);
        setStatusMessage(`Processing face swap... (${n}/60)`);
      });

      setResultVideoUrl(resultUrl);
      setStatusMessage('Done!');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong');
      setStatusMessage('');
    } finally {
      setIsLoading(false);
    }
  }

  async function saveToLibrary() {
    if (!resultVideoUrl) return;
    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== 'granted') {
      setError('Media library permission is required to save.');
      return;
    }
    await MediaLibrary.saveToLibraryAsync(resultVideoUrl);
    setStatusMessage('Saved!');
  }

  const canSwap = targetVideo !== null && faceImage !== null && !isLoading;

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>AI Face Swap</Text>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>Hermes · Expo Router v4</Text>
        </View>

        {/* Target Video */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Target Video</Text>
          <TouchableOpacity style={styles.button} onPress={pickVideo} activeOpacity={0.8}>
            <Text style={styles.buttonText}>Choose Video</Text>
          </TouchableOpacity>
          {targetVideo && (
            <Text style={styles.selectedLabel}>Video selected ✓</Text>
          )}
        </View>

        {/* Face Image */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Source Face</Text>
          <TouchableOpacity style={styles.button} onPress={pickFaceImage} activeOpacity={0.8}>
            <Text style={styles.buttonText}>Choose Image</Text>
          </TouchableOpacity>
          {faceImage && (
            <Image source={{ uri: faceImage.uri }} style={styles.facePreview} />
          )}
        </View>

        {/* Error */}
        {error && <Text style={styles.errorText}>{error}</Text>}

        {/* Swap Button */}
        <TouchableOpacity
          style={[styles.swapButton, !canSwap && styles.swapButtonDisabled]}
          onPress={handleSwap}
          disabled={!canSwap}
          activeOpacity={0.8}
        >
          <Text style={styles.swapButtonText}>Swap Faces</Text>
        </TouchableOpacity>

        {isLoading && <ActivityIndicator size="small" color="#111" />}
        {statusMessage.length > 0 && (
          <Text style={styles.statusText}>{statusMessage}</Text>
        )}

        {/* Result Video */}
        {resultVideoUrl && (
          <View style={styles.resultSection}>
            <VideoView
              player={player}
              style={styles.resultVideo}
              contentFit="contain"
              nativeControls
            />
            <TouchableOpacity style={styles.button} onPress={saveToLibrary} activeOpacity={0.8}>
              <Text style={styles.buttonText}>Save to Camera Roll</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9f9',
  },
  scroll: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    gap: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
    color: '#111',
    letterSpacing: -0.5,
  },
  badge: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 99,
    backgroundColor: '#111',
  },
  badgeText: {
    fontSize: 12,
    color: '#fff',
    fontWeight: '500',
  },
  section: {
    width: '100%',
    gap: 10,
    alignItems: 'flex-start',
  },
  sectionLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  button: {
    backgroundColor: '#111',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  selectedLabel: {
    fontSize: 13,
    color: '#2a9d2a',
    fontWeight: '500',
  },
  facePreview: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: '#ddd',
  },
  errorText: {
    color: '#d00',
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
  },
  swapButton: {
    marginTop: 8,
    width: '100%',
    backgroundColor: '#111',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  swapButtonDisabled: {
    backgroundColor: '#999',
  },
  swapButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  statusText: {
    fontSize: 13,
    color: '#555',
    fontWeight: '500',
  },
  resultSection: {
    width: '100%',
    gap: 12,
    alignItems: 'flex-start',
  },
  resultVideo: {
    width: '100%',
    height: 300,
    borderRadius: 10,
    backgroundColor: '#000',
  },
});
