export type AppMode = 'dev' | 'release';

export function getRuntimeConfig() {
  return typeof window !== 'undefined' ? window.CROSSWORDS_CONFIG : undefined;
}

export function resolveAppMode(): AppMode {
  const runtimeMode = getRuntimeConfig?.()?.APP_MODE;
  if (runtimeMode === 'dev' || runtimeMode === 'release') {
    return runtimeMode;
  }

  const buildMode = import.meta.env.VITE_APP_MODE;
  return buildMode === 'dev' ? 'dev' : 'release';
}
