declare global {
  interface Navigator {
    createHandwritingRecognizer?: (
      options: { languages: string[] }
    ) => Promise<{
      getPrediction: (strokes: unknown) => Promise<unknown>;
    }>;
  }
}

export {};
