import { describe, it, expect, vi, beforeEach } from "vitest";
import { HFTransformersjsCompletionLanguageModelOpenAICompatible } from "./hf-transformersjs-completion-language-model-openai-compatible";
import { pipeline, TextStreamer } from "@huggingface/transformers";
import { mapOpenAICompatibleFinishReason } from './map-openai-compatible-finish-reason';

// Mock the mapOpenAICompatibleFinishReason function
vi.mock('./map-openai-compatible-finish-reason', () => ({
  mapOpenAICompatibleFinishReason: vi.fn().mockReturnValue("stop")
}));

// --- MOCK THE TRANSFORMERS API ---
// Re-use the same mock as before.
vi.mock("@huggingface/transformers", () => {
  return {
    pipeline: vi.fn(),
    TextStreamer: class {
      tokenizer: any;
      callback_function: (token: string) => void;
      constructor(tokenizer: any, options: { callback_function: (token: string) => void; skip_prompt: boolean }) {
        this.tokenizer = tokenizer;
        this.callback_function = options.callback_function;
      }
    },
  };
});

describe("HFTransformersjsCompletionLanguageModelOpenAICompatible", () => {
  const fakePipelineCompletion = async (prompt: string, generationOptions: any) => {
    // Simply echo the prompt plus an appended generated text.
    return [{ generated_text: prompt + " (completed)", finish_reason: "length" }];
  };

  const fakePipelineStreaming = async (prompt: string, generationOptions: any) => {
    // Simulate streaming tokens for completions.
    if (generationOptions.streamer && typeof generationOptions.streamer.callback_function === "function") {
      const tokens = ["This ", "is ", "completed."];
      for (const token of tokens) {
        generationOptions.streamer.callback_function(token);
      }
    }
    return [{ generated_text: "This is completed.", finish_reason: "length" }];
  };

  beforeEach(() => {
    (pipeline as any).mockReset();
    (mapOpenAICompatibleFinishReason as any).mockReset();
    (mapOpenAICompatibleFinishReason as any).mockReturnValue("length");
  });

  it("should generate text with doGenerate using a plain text prompt", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineCompletion);

    const model = new HFTransformersjsCompletionLanguageModelOpenAICompatible(
        "completion-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );
    const result = await model.doGenerate({
      prompt: "Test prompt",
      maxTokens: 150,
      temperature: 0.3,
    });
    expect(result.text).toBe("Test prompt (completed)");
    expect(result.finishReason).toBe("length");
    expect(mapOpenAICompatibleFinishReason).toHaveBeenCalledWith("length");

    const expectedPayload = {
      prompt: "Test prompt",
      max_new_tokens: 150,
      do_sample: true, // temperature (0.3) > 0 so true
      temperature: 0.3,
    };
    expect(result.request.body).toBe(JSON.stringify(expectedPayload));

    // Check that token counts are estimated
    expect(result.usage.promptTokens).toBeGreaterThan(0);
    expect(result.usage.completionTokens).toBeGreaterThan(0);
  });

  it("should stream tokens with doStream", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineStreaming);

    const model = new HFTransformersjsCompletionLanguageModelOpenAICompatible(
        "completion-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const chunks: any[] = [];
    const { stream } = await model.doStream({
      prompt: "Test prompt",
      maxTokens: 150,
      temperature: 0.3,
    });

    const reader = stream.getReader();
    let done = false;

    while (!done) {
      const { value, done: isDone } = await reader.read();
      if (isDone) {
        done = true;
      } else {
        chunks.push(value);
      }
    }

    // Check that we received the expected tokens and a finish event
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks.some(chunk => chunk.type === 'text-delta')).toBe(true);
    expect(chunks.some(chunk => chunk.type === 'finish')).toBe(true);

    // Verify the text chunks match what we expect
    const textChunks = chunks
        .filter(chunk => chunk.type === 'text-delta')
        .map(chunk => chunk.textDelta);

    expect(textChunks).toEqual(["This ", "is ", "completed."]);
  });

  it("should handle message-based prompts", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineCompletion);

    const model = new HFTransformersjsCompletionLanguageModelOpenAICompatible(
        "completion-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const result = await model.doGenerate({
      prompt: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
        { role: "user", content: "How are you?" }
      ],
      maxTokens: 150,
      temperature: 0.3,
    });

    // The prompt should be the messages concatenated with newlines
    expect(result.text).toBe("Hello\nHi there\nHow are you? (completed)");
  });

  it("should handle complex message-based prompts with array content", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineCompletion);

    const model = new HFTransformersjsCompletionLanguageModelOpenAICompatible(
        "completion-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            { type: "text", text: "Hello" },
            { type: "text", text: "World" }
          ]
        }
      ],
      maxTokens: 150,
      temperature: 0.3,
    });

    // The text parts should be joined with spaces
    expect(result.text).toBe("Hello World (completed)");
  });

  it("should provide correct response structure from doGenerate", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineCompletion);

    const model = new HFTransformersjsCompletionLanguageModelOpenAICompatible(
        "completion-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const result = await model.doGenerate({
      prompt: "Test prompt",
      maxTokens: 150,
      temperature: 0,
    });

    expect(result).toHaveProperty('text');
    expect(result).toHaveProperty('finishReason');
    expect(result).toHaveProperty('usage');
    expect(result).toHaveProperty('rawCall');
    expect(result).toHaveProperty('rawResponse');
    expect(result).toHaveProperty('request');

    expect(result.usage).toHaveProperty('promptTokens');
    expect(result.usage).toHaveProperty('completionTokens');
  });
});