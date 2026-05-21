declare module 'onnxruntime-web' {
  export const env: {
    wasm?: {
      proxy?: boolean;
      numThreads?: number;
      wasmPaths?: string | Record<string, string>;
    };
  };

  export class Tensor {
    constructor(type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]);
    data: ArrayLike<number>;
    dims: number[];
    type: string;
  }

  export class InferenceSession {
    inputNames: string[];
    outputNames: string[];
    static create(model: string | ArrayBuffer | Uint8Array): Promise<InferenceSession>;
    run(feeds: Record<string, Tensor>): Promise<Record<string, { data: ArrayLike<number> }>>;
  }
}
