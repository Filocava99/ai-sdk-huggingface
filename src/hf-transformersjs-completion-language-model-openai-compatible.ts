import { pipeline, TextStreamer } from "@huggingface/transformers";
import { LanguageModelV1, LanguageModelV1FinishReason } from '@ai-sdk/provider';
import { mapOpenAICompatibleFinishReason } from './map-openai-compatible-finish-reason';

export class HFTransformersjsCompletionLanguageModelOpenAICompatible implements LanguageModelV1 {
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

  private async ensurePipeline() {
    if (!this.pipelineInstance) {
      this.pipelineInstance = await pipeline("text-generation", this.modelId, {});
    }
    return this.pipelineInstance;
  }

  async doGenerate(options: { prompt: string; maxTokens?: number; temperature?: number; headers?: Record<string, string>; abortSignal?: AbortSignal }): Promise<{ text: string; finishReason: LanguageModelV1FinishReason; rawResponse: unknown; request: { body: string } }> {
    const pn = await this.ensurePipeline();
    const generationOptions = {
      max_new_tokens: options.maxTokens ?? 512,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
    };

    const result = await pn(options.prompt, generationOptions);
    return {
      text: result[0].generated_text,
      finishReason: mapOpenAICompatibleFinishReason(result[0].finish_reason),
      rawResponse: result,
      request: { body: JSON.stringify({ prompt: options.prompt, ...generationOptions }) },
    };
  }

  async doStream(options: { prompt: string; maxTokens?: number; temperature?: number; headers?: Record<string, string>; abortSignal?: AbortSignal; onToken: (token: string) => void }): Promise<{ rawResponse: unknown; request: { body: string } }> {
    const pn = await this.ensurePipeline();
    const streamer = new TextStreamer(pn.tokenizer, {
      skip_prompt: true,
      callback_function: options.onToken,
    });

    const generationOptions = {
      max_new_tokens: options.maxTokens ?? 512,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
      streamer,
    };

    const result = await pn(options.prompt, generationOptions);
    return {
      rawResponse: result,
      request: { body: JSON.stringify({ prompt: options.prompt, ...generationOptions }) },
    };
  }
}
