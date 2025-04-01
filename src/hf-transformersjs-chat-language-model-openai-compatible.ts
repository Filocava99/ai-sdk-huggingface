import {pipeline, TextGenerationOutput, TextGenerationPipeline, TextStreamer} from "@huggingface/transformers";
import {
  LanguageModelV1,
  LanguageModelV1CallOptions,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1FunctionToolCall,
  LanguageModelV1LogProbs,
  LanguageModelV1Prompt,
  LanguageModelV1ProviderMetadata,
  LanguageModelV1Source,
  LanguageModelV1StreamPart,
  LanguageModelV1TextPart
} from '@ai-sdk/provider';

export class HFTransformersjsChatLanguageModelOpenAICompatible implements LanguageModelV1 {
  readonly specificationVersion = 'v1';
  readonly modelId: string;
  readonly settings: Record<string, unknown>;
  readonly provider: string;
  readonly defaultObjectGenerationMode = 'json';

  // @ts-ignore
  private config: { provider: string; apiKey?: string };
  private pipelineInstance?: TextGenerationPipeline;

  constructor(modelId: string, settings: Record<string, unknown>, config: { provider: string; apiKey?: string }) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.provider = config.provider;
  }

  private async ensurePipeline() {
    if (!this.pipelineInstance) {
      this.pipelineInstance = await pipeline("text-generation", this.modelId, {}) as TextGenerationPipeline;
    }
    return this.pipelineInstance;
  }

  doGenerate(options: LanguageModelV1CallOptions): PromiseLike<{
    text?: string;
    reasoning?: string | Array<{
      type: 'text';
      text: string;
      signature?: string;
    } | {
      type: 'redacted';
      data: string;
    }>;
    toolCalls?: Array<LanguageModelV1FunctionToolCall>;
    finishReason: LanguageModelV1FinishReason;
    usage: {
      promptTokens: number;
      completionTokens: number;
    };
    rawCall: {
      rawPrompt: unknown;
      rawSettings: Record<string, unknown>;
    };
    rawResponse?: {
      headers?: Record<string, string>;
      body?: unknown;
    };
    request?: {
      body?: string;
    };
    response?: {
      id?: string;
      timestamp?: Date;
      modelId?: string;
    };
    warnings?: LanguageModelV1CallWarning[];
    providerMetadata?: LanguageModelV1ProviderMetadata;
    sources?: LanguageModelV1Source[];
    logprobs?: LanguageModelV1LogProbs;
  }> {
    return this.doGenerateImpl(options);
  }

  private async doGenerateImpl(options: LanguageModelV1CallOptions): Promise<{
    text?: string;
    reasoning?: string | Array<{
      type: 'text';
      text: string;
      signature?: string;
    } | {
      type: 'redacted';
      data: string;
    }>;
    toolCalls?: Array<LanguageModelV1FunctionToolCall>;
    finishReason: LanguageModelV1FinishReason;
    usage: {
      promptTokens: number;
      completionTokens: number;
    };
    rawCall: {
      rawPrompt: unknown;
      rawSettings: Record<string, unknown>;
    };
    rawResponse?: {
      headers?: Record<string, string>;
      body?: unknown;
    };
    request?: {
      body?: string;
    };
    response?: {
      id?: string;
      timestamp?: Date;
      modelId?: string;
    };
    warnings?: LanguageModelV1CallWarning[];
    providerMetadata?: LanguageModelV1ProviderMetadata;
    sources?: LanguageModelV1Source[];
    logprobs?: LanguageModelV1LogProbs;
  }> {
    const pn = await this.ensurePipeline();
    const promptText = this.convertPromptToString(options.prompt);

    const generationOptions = {
      max_new_tokens: options.maxTokens ?? 512,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
    };

    const res = await pn(promptText, generationOptions);
    const result: TextGenerationOutput[] = Array.isArray(res) ? res as TextGenerationOutput[] : [res];

    // Estimate token counts (basic approximation)
    const promptTokens = Math.ceil(promptText.length / 4); // Very rough estimate
    const completionTokens = Math.ceil(result[0][-1].generated_text.toString().length / 4); // Very rough estimate

    return {
      text: result[0][-1].generated_text.toString(),
      finishReason: "stop" as LanguageModelV1FinishReason,
      usage: {
        promptTokens,
        completionTokens
      },
      rawCall: {
        rawPrompt: promptText,
        rawSettings: generationOptions
      },
      rawResponse: {
        body: result
      },
      request: {
        body: JSON.stringify({ prompt: promptText, ...generationOptions })
      }
    };
  }

  doStream(options: LanguageModelV1CallOptions): PromiseLike<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: {
      rawPrompt: unknown;
      rawSettings: Record<string, unknown>;
    };
    rawResponse?: {
      headers?: Record<string, string> | undefined;
    } | undefined;
    request?: { body: string };
    warnings?: LanguageModelV1CallWarning[];
  }> {
    return this.doStreamImplementation(options);
  }

  private async doStreamImplementation(options: LanguageModelV1CallOptions): Promise<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: {
      rawPrompt: unknown;
      rawSettings: Record<string, unknown>;
    };
    rawResponse?: {
      headers?: Record<string, string> | undefined;
    } | undefined;
    request?: { body: string };
    warnings?: LanguageModelV1CallWarning[];
  }> {
    const pn = await this.ensurePipeline();
    const promptText = this.convertPromptToString(options.prompt);

    const generationOptions = {
      max_new_tokens: options.maxTokens ?? 512,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
    };

    // Create a ReadableStream to return tokens
    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      start: async (controller) => {
        try {
          // Create a callback function that writes to the stream
          const tokenCallback = (token: string) => {
            if (token) {
              controller.enqueue({ type: 'text-delta', textDelta: token });
            }
          };

          // Create a TextStreamer with our callback
          const streamer = new TextStreamer(pn.tokenizer, {
            skip_prompt: true,
            callback_function: tokenCallback,
          });

          // Add streamer to generation options
          const streamerOptions = {
            ...generationOptions,
            streamer,
          };

          // Run the generation with streaming
          const result = await pn(promptText, streamerOptions);

          // Signal completion
          controller.close();

          // Store the result for returning later
          return result;
        } catch (error) {
          controller.error(error);
        }
      }
    });

    return {
      stream,
      rawCall: {
        rawPrompt: promptText,
        rawSettings: generationOptions
      },
      rawResponse: {
        headers: {}
      },
      request: { body: JSON.stringify({ prompt: promptText, ...generationOptions }) }
    };
  }

  // Helper method to convert LanguageModelV1Prompt to string
  private convertPromptToString(prompt: LanguageModelV1Prompt): string {
    if (typeof prompt === 'string') {
      return prompt;
    }

    // Handle array of messages
    if (Array.isArray(prompt)) {
      return prompt.map(message => {
        // Check if content is an array of parts
        if (Array.isArray(message.content)) {
          // Extract text parts only
          return message.content
              .filter((part): part is LanguageModelV1TextPart => 'text' in part)
              .map(part => part.text)
              .join(" ");
        }
        // Legacy format with string content
        else if (typeof message.content === 'string') {
          return `${message.role}: ${message.content}`;
        }
        return "";
      }).join("\n");
    }

    return "";
  }
}