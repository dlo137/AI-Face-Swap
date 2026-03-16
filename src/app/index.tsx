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
const BUCKET_NAME = 'faceswap-uploads';
const MAX_VIDEO_BYTES = 10 * 1024 * 1024; // 10 MB

async function uploadToSupabase(
  fileUri: string,
  fileName: string,
  mimeType: string,
): Promise<string> {
  const response = await fetch(fileUri);
  const blob = await response.blob();

  const res = await fetch(
    `${SUPABASE_URL}/storage/v1/object/${BUCKET_NAME}/${fileName}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': mimeType,
      },
      body: blob,
    },
  );

  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }

  return `${SUPABASE_URL}/storage/v1/object/public/${BUCKET_NAME}/${fileName}`;
}

function uniqueName(ext: string): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}.${ext}`;
}

async function submitFaceSwap(videoUrl: string, imageUrl: string): Promise<string> {
  const res = await fetch('https://api.piapi.ai/api/v1/task', {
    method: 'POST',
    headers: {
      'x-api-key': PIAPI_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'Qubico/video-toolkit',
      task_type: 'face-swap',
      input: {
        swap_image: imageUrl,
        target_video: videoUrl,
        swap_faces_index: '0',
        target_faces_index: '0',
      },
    }),
  });

  const data = await res.json();
  const taskId: string | undefined = data?.data?.task_id;
  if (!taskId) {
    throw new Error('No task_id returned from PiAPI');
  }
  return taskId;
}

async function pollForResult(
  taskId: string,
  onAttempt: (n: number) => void,
): Promise<string> {
  const MAX_ATTEMPTS = 30;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    onAttempt(attempt);
    const res = await fetch(`https://api.piapi.ai/api/v1/task/${taskId}`, {
      headers: { 'x-api-key': PIAPI_KEY },
    });
    const data = await res.json();
    const status: string = data?.data?.status;

    if (status === 'completed') {
      const videoUrl: string | undefined = data?.data?.output?.video_url;
      if (!videoUrl) throw new Error('No video_url in completed response');
      return videoUrl;
    }
    if (status === 'failed') {
      throw new Error('Face swap failed');
    }

    if (attempt < MAX_ATTEMPTS) {
      await new Promise<void>((resolve) => setTimeout(resolve, 4000));
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

  const player = useVideoPlayer(resultVideoUrl ?? '', (p) => {
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
      setStatusMessage('Uploading video...');
      const videoExt = targetVideo.uri.split('.').pop() ?? 'mp4';
      const videoUrl = await uploadToSupabase(
        targetVideo.uri,
        uniqueName(videoExt),
        targetVideo.mimeType ?? 'video/mp4',
      );

      setStatusMessage('Uploading image...');
      const imageExt = faceImage.uri.split('.').pop() ?? 'jpg';
      const imageUrl = await uploadToSupabase(
        faceImage.uri,
        uniqueName(imageExt),
        faceImage.mimeType ?? 'image/jpeg',
      );

      setStatusMessage('Submitting to PiAPI...');
      const taskId = await submitFaceSwap(videoUrl, imageUrl);

      setStatusMessage('Processing... (attempt 1)');
      const resultUrl = await pollForResult(taskId, (n) => {
        setPollAttempt(n);
        setStatusMessage(`Processing... (attempt ${n})`);
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
