// src/hf-transformersjs-chat-language-model.ts
import {pipeline, TextGenerationOutput, TextGenerationPipeline, TextStreamer} from "@huggingface/transformers";
import {
  LanguageModelV1,
  LanguageModelV1CallOptions,
  LanguageModelV1CallWarning, LanguageModelV1FinishReason, LanguageModelV1FunctionToolCall, LanguageModelV1LogProbs,
  LanguageModelV1Prompt, LanguageModelV1ProviderMetadata, LanguageModelV1Source,
  LanguageModelV1StreamPart,
  LanguageModelV1TextPart
} from '@ai-sdk/provider';

export class HFTransformersjsChatLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1';
  readonly modelId: string;
  readonly settings: Record<string, unknown>;
  readonly provider: string;
  readonly defaultObjectGenerationMode = 'json';

  // Common stop sequences for most chat models
  private readonly defaultStopSequences = ["</s>", "<|endoftext|>", "Human:", "Assistant:", "User:"];
  
  // @ts-ignore
  private config: { provider: string; apiKey?: string };
  // Hold our pipeline instance once loaded.
  private pipelineInstance?: TextGenerationPipeline;

  constructor(modelId: string, settings: Record<string, unknown>, config: { provider: string; apiKey?: string }) {
    console.log(`[DEBUG] Initializing HFTransformersjsChatLanguageModel with modelId: ${modelId}`);
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.provider = config.provider;
  }

  // Lazy-load the text generation pipeline.
  private async ensurePipeline() {
    console.log('[DEBUG] ensurePipeline() called');
    if (!this.pipelineInstance) {
      console.log(`[DEBUG] Pipeline not initialized, loading model: ${this.modelId}`);
      try {
        // @ts-ignore
        this.pipelineInstance = await pipeline("text-generation", this.modelId, {
          progress_callback: (progress) => {
            console.log(`[DEBUG] Model loading progress: ${JSON.stringify(progress)}%`);
          }
        }) as TextGenerationPipeline;
        console.log(`[DEBUG] Pipeline successfully loaded`);
      } catch (error) {
        console.error(`[ERROR] Failed to load pipeline:`, error);
        throw error;
      }
    } else {
      console.log('[DEBUG] Using existing pipeline instance');
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
    console.log('[DEBUG] doGenerate() called');
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
    console.log('[DEBUG] doGenerateImpl() called with options:', JSON.stringify(options, null, 2));

    console.log('[DEBUG] Getting pipeline...');
    const pn = await this.ensurePipeline();
    console.log('[DEBUG] Pipeline retrieved successfully');

    console.log('[DEBUG] Converting prompt to string');
    const promptText = this.convertPromptToString(options.prompt);
    console.log('[DEBUG] Prompt converted to string:', promptText.substring(0, 100) + (promptText.length > 100 ? '...' : ''));

    // FIX 1: Add stop sequences
    const customStopSequences = options.stopSequences || [];
    const stopSequences = [...this.defaultStopSequences, ...customStopSequences];
    
    // FIX 3: Limit max tokens if too high
    const maxTokens = Math.min(options.maxTokens ?? 256, 512); // Reduced default and capped at 512
    
    const generationOptions = {
      max_new_tokens: maxTokens,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
      stop_strings: stopSequences,
      top_p: options.topP ?? 0.9,
      repetition_penalty: 1.2,
      early_stopping: true,
    };
    console.log('[DEBUG] Generation options:', generationOptions);

    console.log('[DEBUG] Calling pipeline...');
    try {
      const res = await pn(promptText, generationOptions);
      console.log('[DEBUG] Pipeline result received');
      console.log('[DEBUG] Raw pipeline result:', JSON.stringify(res, null, 2));

      // Process the result properly
      const result: TextGenerationOutput[] = Array.isArray(res) ? res as TextGenerationOutput[] : [res];
      console.log('[DEBUG] Pipeline result processed');

      // FIX 2: Safely access and process the generated text
      if (!result || !result.length) {
        console.error('[ERROR] Empty result from pipeline');
        throw new Error('Empty result from pipeline');
      }
      
      // Safely extract the generated text with proper type checking and error handling
      let generatedText: string;
      try {
        // FIX 2: Correct handling of TextGenerationOutput (which is an array of TextGenerationSingle)
        const firstOutput: TextGenerationOutput = result[0];
        
        // Verify firstOutput is an array as expected for TextGenerationOutput
        if (Array.isArray(firstOutput) && firstOutput.length > 0) {
          // Get the last generation from the array
          const lastGeneration = firstOutput[firstOutput.length - 1];
          
          if (typeof lastGeneration === 'string') {
            generatedText = lastGeneration;
          } else if (lastGeneration && typeof lastGeneration.generated_text === 'string') {
            generatedText = lastGeneration.generated_text;
          } else if (lastGeneration && lastGeneration.generated_text && typeof lastGeneration.generated_text === 'string') {
            generatedText = lastGeneration.generated_text;
          } else {
            console.error('[ERROR] Cannot extract text from generation output:', lastGeneration);
            throw new Error('Unable to extract text from generation output');
          }
        } else {
          console.error('[ERROR] Expected TextGenerationOutput to be an array:', firstOutput);
          throw new Error('Unexpected format for TextGenerationOutput');
        }
        
        // FIX 2: Remove the prompt from the generated text to avoid duplication
        if (generatedText.startsWith(promptText)) {
          generatedText = generatedText.substring(promptText.length).trim();
        }
      } catch (error) {
        console.error('[ERROR] Failed to extract generated text:', error);
        throw new Error('Failed to extract generated text from model response');
      }

      // Estimate token counts (basic approximation)
      const promptTokens = Math.ceil(promptText.length / 4); // Very rough estimate
      const completionTokens = Math.ceil(generatedText.length / 4); // Very rough estimate

      console.log('[DEBUG] Generated text length:', generatedText.length);
      console.log('[DEBUG] First 100 chars of generated text:', generatedText.substring(0, 100) + (generatedText.length > 100 ? '...' : ''));

      return {
        text: generatedText,
        finishReason: "stop" as LanguageModelV1FinishReason, // Assuming normal completion
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
          body: JSON.stringify({prompt: promptText, ...generationOptions})
        }
      };
    } catch (error) {
      console.error('[ERROR] Pipeline execution failed:', error);
      throw error;
    }
  }

  // Helper method to convert LanguageModelV1Prompt to string
  private convertPromptToString(prompt: LanguageModelV1Prompt): string {
    console.log('[DEBUG] convertPromptToString() called with prompt type:', typeof prompt);

    if (typeof prompt === 'string') {
      console.log('[DEBUG] Prompt is a string');
      return prompt;
    }

    // Handle array of messages
    if (Array.isArray(prompt)) {
      console.log('[DEBUG] Prompt is an array of messages with length:', prompt.length);

      return prompt.map(message => {
        console.log('[DEBUG] Processing message with role:', message.role);
        // Check if content is an array of parts
        if (Array.isArray(message.content)) {
          console.log('[DEBUG] Message content is an array of parts with length:', message.content.length);
          // Extract text parts only
          const result = message.content
              .filter((part): part is LanguageModelV1TextPart => 'text' in part)
              .map(part => part.text)
              .join(" ");
          console.log('[DEBUG] Extracted text from parts:', result.substring(0, 50) + (result.length > 50 ? '...' : ''));
          return result;
        }
        // Legacy format with string content
        else if (typeof message.content === 'string') {
          console.log('[DEBUG] Message content is a string');
          const result = `${message.role}: ${message.content}`;
          console.log('[DEBUG] Formatted message:', result.substring(0, 50) + (result.length > 50 ? '...' : ''));
          return result;
        }
        console.log('[DEBUG] Unknown message content format');
        return "";
      }).join("\n");
    }

    console.log('[DEBUG] Unrecognized prompt format, returning empty string');
    return "";
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
    console.log('[DEBUG] doStream() called');
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
    console.log('[DEBUG] doStreamImplementation() called');

    console.log('[DEBUG] Getting pipeline for streaming...');
    const pn = await this.ensurePipeline();
    console.log('[DEBUG] Pipeline for streaming retrieved successfully');

    const promptText = this.convertPromptToString(options.prompt);
    console.log('[DEBUG] Prompt converted to string for streaming');

    // FIX 1: Add stop sequences for streaming too
    const customStopSequences = options.stopSequences || [];
    const stopSequences = [...this.defaultStopSequences, ...customStopSequences];
    
    // FIX 3: Limit max tokens if too high
    const maxTokens = Math.min(options.maxTokens ?? 256, 512); // Reduced default and capped at 512

    const generationOptions = {
      max_new_tokens: maxTokens,
      do_sample: (options.temperature ?? 0) > 0,
      temperature: options.temperature ?? 0,
      // FIX 1: Explicitly set stop sequences for streaming
      stop_strings: stopSequences,
      // FIX 4: Add more controls for generation in streaming
      top_p: options.topP ?? 0.9,
      repetition_penalty: 1.2, // FIX 5: Add repetition penalty
      early_stopping: true,    // FIX 5: Enable early stopping
    };
    console.log('[DEBUG] Streaming generation options:', generationOptions);

    // Track repetition for streaming
    let lastTokens: string[] = [];
    const maxTrackedTokens = 50;
    const repetitionThreshold = 10;

    // Create a ReadableStream to return tokens
    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      start: async (controller) => {
        console.log('[DEBUG] Stream start callback executing');
        try {
          // Create a callback function that writes to the stream
          const tokenCallback = (token: string) => {
            if (token) {
              console.log(`[DEBUG] Token received: "${token}"`);
              
              // FIX 5: Check for repetition in streaming
              lastTokens.push(token);
              if (lastTokens.length > maxTrackedTokens) {
                lastTokens.shift();
              }
              
              // Check if we should stop due to repetition
              let shouldStop = false;
              
              // Check for stop sequences in concatenated tokens
              const recentText = lastTokens.join('');
              for (const stopSeq of stopSequences) {
                if (recentText.includes(stopSeq)) {
                  console.log(`[DEBUG] Found stop sequence in streaming: "${stopSeq}"`);
                  shouldStop = true;
                  break;
                }
              }
              
              // Simple repetition detection for streaming
              if (lastTokens.length >= repetitionThreshold) {
                const firstHalf = lastTokens.slice(0, repetitionThreshold/2).join('');
                const secondHalf = lastTokens.slice(repetitionThreshold/2, repetitionThreshold).join('');
                if (firstHalf === secondHalf) {
                  console.log('[DEBUG] Detected token repetition in streaming, stopping');
                  shouldStop = true;
                }
              }
              
              if (!shouldStop) {
                controller.enqueue({ type: 'text-delta', textDelta: token });
              } else {
                // Signal stop to the streaming mechanism if possible
                return false;
              }
            }
            return true;
          };

          // Create a TextStreamer with our callback
          console.log('[DEBUG] Creating TextStreamer');
          const streamer = new TextStreamer(pn.tokenizer, {
            skip_prompt: true,
            callback_function: tokenCallback,
          });
          console.log('[DEBUG] TextStreamer created');

          // Add streamer to generation options
          const streamerOptions = {
            ...generationOptions,
            streamer,
          };

          // Run the generation with streaming
          console.log('[DEBUG] Starting streaming generation');
          const result = await pn(promptText, streamerOptions);
          console.log('[DEBUG] Streaming generation completed');

          // Signal completion
          console.log('[DEBUG] Closing stream controller');
          controller.close();

          // Store the result for returning later
          return result;
        } catch (error) {
          console.error('[ERROR] Error in stream processing:', error);
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
}