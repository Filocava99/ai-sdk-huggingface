import { pipeline } from "@huggingface/transformers";
import { EmbeddingModelV1, TooManyEmbeddingValuesForCallError } from '@ai-sdk/provider';

// Define the embedding type expected by the interface
type EmbeddingModelV1Embedding = number[];

export class HFTransformersjsEmbeddingModel implements EmbeddingModelV1<string> {
  readonly specificationVersion = 'v1';
  readonly modelId: string;
  readonly settings: Record<string, unknown>;
  readonly provider: string;
  readonly supportsParallelCalls: boolean = true;

  private config: { provider: string; apiKey?: string };
  private pipelineInstance?: Awaited<ReturnType<typeof pipeline>>;

  constructor(modelId: string, settings: Record<string, unknown>, config: { provider: string; apiKey?: string }) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.provider = config.provider;
  }

  // Define maxEmbeddingsPerCall or use a default:
  get maxEmbeddingsPerCall(): number {
    return 2048;
  }

  private async ensurePipeline() {
    if (!this.pipelineInstance) {
      // For feature-extraction pipeline, specify the proper config
      this.pipelineInstance = await pipeline("feature-extraction", this.modelId, {
        revision: "main",
        // quantized: false
      });
    }
    // Return the pipeline instance, properly typed
    return this.pipelineInstance as unknown as (input: string, options: any) => Promise<unknown>;
  }

  // Helper function to safely extract numeric arrays from pipeline outputs
  private extractVectorsFromOutput(output: unknown): number[][] {
    // Handle empty output
    if (!output) return [[]];

    try {
      // Case 1: If it's a Tensor with a data property
      if (output && typeof output === 'object' && 'data' in output) {
        const data = (output as any).data;
        if (Array.isArray(data)) {
          if (data.length === 0) return [[]];
          return Array.isArray(data[0]) ? data : [data];
        }
      }

      // Case 2: If it's an array
      if (Array.isArray(output)) {
        if (output.length === 0) return [[]];

        // Check if it contains numeric arrays
        if (Array.isArray(output[0]) &&
            output[0].length > 0 &&
            typeof output[0][0] === 'number') {
          return output as number[][];
        }

        // Or if it's a flat numeric array
        if (typeof output[0] === 'number') {
          return [output as number[]];
        }
      }

      // Case 3: If it has a toArray method (common for tensors)
      if (output && typeof output === 'object' && 'toArray' in output &&
          typeof (output as any).toArray === 'function') {
        const arrayData = (output as any).toArray();
        if (Array.isArray(arrayData)) {
          if (arrayData.length === 0) return [[]];
          return Array.isArray(arrayData[0]) ? arrayData : [arrayData];
        }
      }

      // Case 4: If it's a typed array or array-like object
      if (output && typeof output === 'object' && 'length' in output) {
        try {
          const arrayData = Array.from(output as any);
          if (arrayData.length === 0) return [[]];
          return Array.isArray(arrayData[0]) ? arrayData as number[][] : [arrayData as number[]];
        } catch (e) {
          console.error("Failed to convert array-like object:", e);
        }
      }

      // Case 5: For models that return objects with hidden vector data
      // Some models may store vectors in specific properties
      if (output && typeof output === 'object') {
        for (const key of ['embedding', 'embeddings', 'vector', 'vectors', 'features']) {
          if (key in output && Array.isArray((output as any)[key])) {
            const vectorData = (output as any)[key];
            if (vectorData.length === 0) return [[]];
            return Array.isArray(vectorData[0]) ? vectorData : [vectorData];
          }
        }
      }

      // Last resort: try to stringify and parse to extract any numeric arrays
      console.warn("Using fallback extraction method for pipeline output:", output);
      return [[]];
    } catch (error) {
      console.error("Error extracting vectors from pipeline output:", error);
      return [[]];
    }
  }

  async doEmbed(options: {
    values: string[];
    headers?: Record<string, string>;
    abortSignal?: AbortSignal
  }): Promise<{
    embeddings: EmbeddingModelV1Embedding[];
    usage?: { tokens: number };
    rawResponse?: { headers?: Record<string, string> }
  }> {
    if (options.values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.config.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values: options.values,
      });
    }

    const extractor = await this.ensurePipeline();
    const embeddings: EmbeddingModelV1Embedding[] = [];

    for (const input of options.values) {
      try {
        // Call the extractor with the input and empty options object to satisfy the type requirements
        const output = await extractor(input, {});

        // Extract properly formatted vectors from the output
        const tokenVectors = this.extractVectorsFromOutput(output);

        // Average the token vectors to obtain one embedding vector for the input
        const numTokens = tokenVectors.length;
        if (numTokens === 0 || (numTokens === 1 && tokenVectors[0].length === 0)) {
          embeddings.push([]);
          continue;
        }

        const dim = tokenVectors[0].length;
        const avg = new Array(dim).fill(0);

        for (const vec of tokenVectors) {
          if (vec && vec.length === dim) {
            for (let i = 0; i < dim; i++) {
              avg[i] += vec[i];
            }
          }
        }

        for (let i = 0; i < dim; i++) {
          avg[i] /= numTokens;
        }

        embeddings.push(avg);
      } catch (error) {
        console.error(`Error embedding input "${input.substring(0, 50)}...":`, error);
        // Return an empty vector on error to maintain array alignment with inputs
        embeddings.push([]);
      }
    }

    return {
      embeddings,
      usage: { tokens: options.values.length },
      rawResponse: { headers: options.headers }
    };
  }
}