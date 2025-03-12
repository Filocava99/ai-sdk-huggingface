import { pipeline } from "@huggingface/transformers";
import { EmbeddingModelV1, TooManyEmbeddingValuesForCallError } from '@ai-sdk/provider';

export class HFTransformersjsEmbeddingModel implements EmbeddingModelV1<string> {
  readonly specificationVersion = 'v1';
  readonly modelId: string;
  readonly settings: Record<string, unknown>;

  private config: { provider: string; apiKey?: string };
  private pipelineInstance?: Awaited<ReturnType<typeof pipeline>>;

  constructor(modelId: string, settings: Record<string, unknown>, config: { provider: string; apiKey?: string }) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  // Define maxEmbeddingsPerCall or use a default:
  get maxEmbeddingsPerCall(): number {
    return 2048;
  }

  private async ensurePipeline() {
    if (!this.pipelineInstance) {
      this.pipelineInstance = await pipeline("feature-extraction", this.modelId, {});
    }
    return this.pipelineInstance;
  }

  async doEmbed(options: { values: string[]; headers?: Record<string, string>; abortSignal?: AbortSignal }): Promise<{ embeddings: number[][]; usage?: { tokens: number }; rawResponse: unknown }> {
    if (options.values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.config.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values: options.values,
      });
    }
    const extractor = await this.ensurePipeline();
    const embeddings = [];
    for (const input of options.values) {
      // feature-extraction returns a 2D array: one array per token.
      const tokenVectors: number[][] = await extractor(input, {});  
      // Average the token vectors to obtain one embedding vector for the input.
      const numTokens = tokenVectors.length;
      if (numTokens === 0) {
        embeddings.push([]);
        continue;
      }
      const dim = tokenVectors[0].length;
      const avg = new Array(dim).fill(0);
      for (const vec of tokenVectors) {
        for (let i = 0; i < dim; i++) {
          avg[i] += vec[i];
        }
      }
      for (let i = 0; i < dim; i++) {
        avg[i] /= numTokens;
      }
      embeddings.push(avg);
    }

    return {
      embeddings,
      usage: { tokens: options.values.length },
      rawResponse: embeddings,
    };
  }
}
