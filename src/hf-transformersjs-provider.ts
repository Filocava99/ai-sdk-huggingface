import {
  EmbeddingModelV1,
  LanguageModelV1,
  ProviderV1,
} from '@ai-sdk/provider';
// Import your provider-specific language model implementations:
import { HFTransformersjsChatLanguageModel } from './hf-transformersjs-chat-language-model';
import { HFTransformersjsCompletionLanguageModel } from './hf-transformersjs-completion-language-model';
import { HFTransformersjsEmbeddingModel } from './hf-transformersjs-embedding-model';

export interface HFTransformersjsProvider<
    CHAT_MODEL_IDS extends string = string,
    COMPLETION_MODEL_IDS extends string = string,
    EMBEDDING_MODEL_IDS extends string = string,
> extends ProviderV1 {
  (
      modelId: CHAT_MODEL_IDS,
      settings?: Record<string, unknown>,
  ): LanguageModelV1;

  languageModel(
      modelId: CHAT_MODEL_IDS,
      settings?: Record<string, unknown>,
  ): LanguageModelV1;

  chatModel(
      modelId: CHAT_MODEL_IDS,
      settings?: Record<string, unknown>,
  ): LanguageModelV1;

  completionModel(
      modelId: COMPLETION_MODEL_IDS,
      settings?: Record<string, unknown>,
  ): LanguageModelV1;

  textEmbeddingModel(
      modelId: EMBEDDING_MODEL_IDS,
      settings?: Record<string, unknown>,
  ): EmbeddingModelV1<string>;
}

export interface HFTransformersjsProviderSettings {
  /**
   * Provider name.
   */
  name: string;

  /**
   * API key for authenticating with HuggingFace, if required.
   */
  apiKey?: string;
}

export function createHFTransformersjs<
    CHAT_MODEL_IDS extends string,
    COMPLETION_MODEL_IDS extends string,
    EMBEDDING_MODEL_IDS extends string,
>(
    options: HFTransformersjsProviderSettings,
): HFTransformersjsProvider<
    CHAT_MODEL_IDS,
    COMPLETION_MODEL_IDS,
    EMBEDDING_MODEL_IDS
> {
  // Create the common configuration for pipelines
  const getCommonModelConfig = () => ({
    provider: options.name,
    apiKey: options.apiKey,
  });

  // Create a chat / text-generation model
  const createChatModel = (
      modelId: CHAT_MODEL_IDS,
      settings: Record<string, unknown> = {},
  ): LanguageModelV1 =>
      new HFTransformersjsChatLanguageModel(modelId, settings, getCommonModelConfig());

  // For legacy or general languageModel access
  const createLanguageModel = (
      modelId: CHAT_MODEL_IDS,
      settings: Record<string, unknown> = {},
  ): LanguageModelV1 => createChatModel(modelId, settings);

  // Create a completion model if it's somehow different
  const createCompletionModel = (
      modelId: COMPLETION_MODEL_IDS,
      settings: Record<string, unknown> = {},
  ): LanguageModelV1 =>
      new HFTransformersjsCompletionLanguageModel(modelId, settings, getCommonModelConfig());

  // Create an embedding model
  const createEmbeddingModel = (
      modelId: EMBEDDING_MODEL_IDS,
      settings: Record<string, unknown> = {},
  ): EmbeddingModelV1<string> =>
      new HFTransformersjsEmbeddingModel(modelId, settings, getCommonModelConfig());

  // The main provider function (returns a default language model)
  const provider = function(
      modelId: CHAT_MODEL_IDS,
      settings?: Record<string, unknown>,
  ): LanguageModelV1 {
    return createLanguageModel(modelId, settings);
  } as unknown as HFTransformersjsProvider<
      CHAT_MODEL_IDS,
      COMPLETION_MODEL_IDS,
      EMBEDDING_MODEL_IDS
  >;

  provider.languageModel = createLanguageModel;
  provider.chatModel = createChatModel;
  provider.completionModel = createCompletionModel;
  provider.textEmbeddingModel = createEmbeddingModel;

  return provider;
}