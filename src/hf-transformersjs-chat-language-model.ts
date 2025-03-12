import { pipeline, TextStreamer } from "@huggingface/transformers";
import { LanguageModelV1, LanguageModelV1StreamPart } from '@ai-sdk/provider';

export class HFTransformersjsChatLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1';
  readonly modelId: string;
  readonly settings: Record<string, unknown>; // adjust type as needed

  private config: { provider: string; apiKey?: string }; // from getCommonModelConfig()
  // Hold our pipeline instance once loaded.
  private pipelineInstance?: Awaited<ReturnType<typeof pipeline>>;

  constructor(modelId: string, settings: Record<string, unknown>, config: { provider: string; apiKey?: string }) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  // Lazy-load the text generation pipeline.
  private async ensurePipeline() {
    if (!this.pipelineInstance) {
      this.pipelineInstance = await pipeline("text-generation", this.modelId, {});
    }
    return this.pipelineInstance;
  }

  async doGenerate(options: { prompt: string | Array<{role: string; content: string}>; maxTokens?: number; temperature?: number; headers?: Record<string, string>; abortSignal?: AbortSignal; }): Promise<{ text: string; rawResponse: unknown; request: { body: string } }> {
    const pn = await this.ensurePipeline();
    // If prompt is given as messages array, combine into a single string.
    const promptText = Array.isArray(options.prompt)
      ? options.prompt.map(m => m.content).join("\n")
      : options.prompt;

    const generationOptions = {
      max_new_tokens: options.maxTokens ?? 512,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
      // add any other options as needed
    };

    const result = await pn(promptText, generationOptions);
    // (Assume result is an array with at least one element.)
    return {
      text: result[0].generated_text,
      rawResponse: result,
      request: { body: JSON.stringify({ prompt: promptText, ...generationOptions }) },
    };
  }

  async doStream(options: { prompt: string | Array<{role: string; content: string}>; maxTokens?: number; temperature?: number; headers?: Record<string, string>; abortSignal?: AbortSignal; onToken: (token: string) => void; }): Promise<{ rawResponse: unknown; request: { body: string } }> {
    const pn = await this.ensurePipeline();
    const promptText = Array.isArray(options.prompt)
      ? options.prompt.map(m => m.content).join("\n")
      : options.prompt;

    // Create a TextStreamer with the passed callback.
    const streamer = new TextStreamer(pn.tokenizer, {
      skip_prompt: true,
      callback_function: options.onToken,
    });

    const generationOptions = {
      max_new_tokens: options.maxTokens ?? 512,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
      streamer, // add streamer for streaming
    };

    // Call pipeline with streamer; expect that tokens are streamed via callback.
    const result = await pn(promptText, generationOptions);
    // The pipelined call completes when the stream ends.
    return {
      rawResponse: result,
      request: { body: JSON.stringify({ prompt: promptText, ...generationOptions }) },
    };
  }
}
