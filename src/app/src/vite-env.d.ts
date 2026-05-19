/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SRV_URL?: string;
  readonly VITE_TRAIN_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface CrosswordsRuntimeConfig {
  MODEL_BASE_URL?: string;
  SRV_URL?: string;
}

interface Window {
  CROSSWORDS_CONFIG?: CrosswordsRuntimeConfig;
}
