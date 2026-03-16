import { createClient } from '@supabase/supabase-js';
import * as SecureStore from 'expo-secure-store';

// ---------------------------------------------------------------------------
// TODO: Replace these placeholders with your real Supabase project credentials.
// Find them in the Supabase dashboard → Project Settings → API.
//
// DO NOT commit real credentials. Store them as EAS secrets:
//   eas secret:create --scope project --name EXPO_PUBLIC_SUPABASE_URL --value "..."
//   eas secret:create --scope project --name EXPO_PUBLIC_SUPABASE_ANON_KEY --value "..."
//
// Then reference them here via process.env.EXPO_PUBLIC_SUPABASE_URL etc.
// ---------------------------------------------------------------------------
const SUPABASE_URL = process.env['EXPO_PUBLIC_SUPABASE_URL'] ?? 'https://your-project-id.supabase.co';
const SUPABASE_ANON_KEY = process.env['EXPO_PUBLIC_SUPABASE_ANON_KEY'] ?? 'your-anon-key-here';

/**
 * SecureStore adapter so Supabase Auth persists sessions in the device
 * keychain (iOS Keychain / Android Keystore) rather than AsyncStorage.
 */
const SecureStoreAdapter = {
  getItem: (key: string): Promise<string | null> =>
    SecureStore.getItemAsync(key),
  setItem: (key: string, value: string): Promise<void> =>
    SecureStore.setItemAsync(key, value),
  removeItem: (key: string): Promise<void> =>
    SecureStore.deleteItemAsync(key),
};

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    storage: SecureStoreAdapter,
    autoRefreshToken: true,
    persistSession: true,
    // Required for React Native — no URL-based OAuth callback
    detectSessionInUrl: false,
  },
});
