import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function HomeScreen() {
  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.content}>
        <Text style={styles.title}>AI Face Swap</Text>
        <Text style={styles.subtitle}>
          Expo SDK 55 · React Native 0.83 · New Architecture
        </Text>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>Hermes · Expo Router v4</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9f9',
  },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    gap: 12,
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
    color: '#111',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 14,
    color: '#555',
    textAlign: 'center',
  },
  badge: {
    marginTop: 8,
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
});
