# AI Face Swap

React Native app built with **Expo SDK 55**, **React Native 0.83**, and **Expo Router v4**.
New Architecture is mandatory and enabled by default in SDK 55.

---

## Stack

| Layer | Package | Version |
|---|---|---|
| Runtime | Expo SDK | ^55.0.0 |
| Framework | React Native | 0.83.x |
| Routing | Expo Router | ^4.0.0 |
| JS Engine | Hermes | default |
| Architecture | New Architecture | enabled (required) |
| Auth / DB client | @supabase/supabase-js | ^2.x |
| Secure storage | expo-secure-store | ^14.x |
| Animations | react-native-reanimated | ~4.x |
| Gestures | react-native-gesture-handler | ~2.22.0 |
| Screens | react-native-screens | ~4.x |
| Safe area | react-native-safe-area-context | ~5.x |

---

## Prerequisites

- Node.js 20 LTS (`node -v` must be ≥ 20.19.4)
- npm 10+ (bundled with Node 20)
- EAS CLI: `npm install -g eas-cli`
- An Expo account: `eas login`
- An EAS project linked: `eas init` (first time only)

---

## Initial setup

```bash
# 1. Install dependencies — use npx expo install, not npm install,
#    for native modules so Expo resolves compatible versions.
npm install

# 2. Install / re-resolve native packages with Expo's version resolver
npx expo install

# 3. (Optional) Type-check
npm run ts:check
```

---

## Running in development

### Expo Go (JS-only, limited native modules)
```bash
npx expo start
```

### Development build (full native — recommended)
Because this project uses `expo-secure-store` and other native modules that
are not supported inside Expo Go, you should use a **development build**.

```bash
# Build a development client (iOS simulator — EAS cloud build)
eas build --platform ios --profile development

# Build a development client (Android APK — EAS cloud build)
eas build --platform android --profile development
```

After the build installs on your device/simulator, start the dev server:
```bash
npx expo start --dev-client
```

---

## ⚠️ Windows — no local Xcode / CocoaPods required

This project is developed on **Windows** and does **not** require a local Mac
environment for building. All iOS and Android builds run through
**EAS Build** on Expo's cloud infrastructure.

| Task | Command | Where it runs |
|---|---|---|
| iOS debug build | `eas build --platform ios --profile development` | EAS cloud (macOS) |
| Android debug APK | `eas build --platform android --profile development` | EAS cloud (Linux) |
| iOS preview build | `eas build --platform ios --profile preview` | EAS cloud (macOS) |
| Android preview APK | `eas build --platform android --profile preview` | EAS cloud (Linux) |
| iOS production | `eas build --platform ios --profile production` | EAS cloud (macOS) |
| Android production AAB | `eas build --platform android --profile production` | EAS cloud (Linux) |

> All build profiles use `"image": "auto"` so EAS automatically selects the
> correct build environment for Expo SDK 55.

---

## Environment variables & secrets

Supabase credentials are **not** committed to source control.
Store them as EAS secrets and reference them via `EXPO_PUBLIC_*` env vars:

```bash
eas secret:create --scope project --name EXPO_PUBLIC_SUPABASE_URL --value "https://xxxx.supabase.co"
eas secret:create --scope project --name EXPO_PUBLIC_SUPABASE_ANON_KEY --value "eyJ..."
```

Then update `src/lib/supabase.ts` — the placeholder values already read from
`process.env.EXPO_PUBLIC_SUPABASE_URL` and `process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY`.

For local development, create a `.env.local` file (already gitignored):
```
EXPO_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

---

## Project structure

```
/
├── src/
│   ├── app/              # Expo Router file-based routes
│   │   ├── _layout.tsx   # Root layout (GestureHandler + SafeArea + Stack)
│   │   └── index.tsx     # Home screen "/"
│   ├── components/       # Shared UI components
│   ├── hooks/            # Custom React hooks
│   └── lib/
│       └── supabase.ts   # Supabase client (configure credentials here)
├── assets/               # Static assets (icon.png, splash-icon.png, etc.)
├── app.json              # Expo config (newArchEnabled: true, jsEngine: hermes)
├── eas.json              # EAS Build profiles (development / preview / production)
├── babel.config.js       # Babel config (reanimated plugin last)
├── tsconfig.json         # TypeScript strict mode + path aliases
└── package.json
```

---

## Required asset files

Place the following PNG files in `/assets/` before building:

| File | Size | Usage |
|---|---|---|
| `icon.png` | 1024×1024 | App icon |
| `splash-icon.png` | 1284×2778 | Splash screen |
| `adaptive-icon.png` | 1024×1024 | Android adaptive icon foreground |
| `favicon.png` | 32×32 | Web favicon |

---

## Adding native packages

Always use `npx expo install` (not `npm install`) for packages with native code
so that Expo's version resolver selects compatible versions:

```bash
npx expo install expo-image expo-camera expo-video
```

---

## Key SDK 55 notes

- **New Architecture is mandatory** — `newArchEnabled: false` is not a valid option in SDK 55+.
- **Hermes is the only supported JS engine** — do not switch to JSC.
- **expo-av is deprecated** — use `expo-video` for video playback.
- **Expo Router v4** ships with SDK 55 and handles all navigation — do not install `@react-navigation/*` packages manually.
- Metro config overrides are not needed for SDK 55 — do not add `metro.config.js` overrides unless a specific package requires it.
