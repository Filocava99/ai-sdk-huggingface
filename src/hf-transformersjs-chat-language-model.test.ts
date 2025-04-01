// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from "vitest";
import { HFTransformersjsChatLanguageModel } from "./hf-transformersjs-chat-language-model";
import { pipeline, TextStreamer } from "@huggingface/transformers";
import {LanguageModelV1CallOptions, LanguageModelV1Prompt, LanguageModelV1TextPart} from "@ai-sdk/provider";

// --- MOCK THE TRANSFORMERS API ---
// We simulate the pipeline() function and the TextStreamer class.
vi.mock("@huggingface/transformers", () => {
  return {
    // Our fake pipeline returns an async function
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

describe("HFTransformersjsChatLanguageModel", () => {
  let mockTextGenerationPipeline;

  const fakePipelineResultNonStreaming = [
    {
      [-1]: { generated_text: "Hello (generated)" }
    }
  ];

  beforeEach(() => {
    // Reset the mock before each test.
    vi.clearAllMocks();

    mockTextGenerationPipeline = vi.fn().mockImplementation(async (prompt, options) => {
      // For streaming mode, use the streamer callbacks if provided
      if (options.streamer && typeof options.streamer.callback_function === "function") {
        const tokens = ["Hello", " ", "world", "!"];
        for (const token of tokens) {
          options.streamer.callback_function(token);
        }
      }

      // Return the appropriate response
      return fakePipelineResultNonStreaming;
    });

    // Mock the pipeline factory function
    (pipeline as any).mockResolvedValue(mockTextGenerationPipeline);
  });

  it("should generate text with doGenerate using a string prompt", async () => {
    const model = new HFTransformersjsChatLanguageModel(
        "test-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const prompt: LanguageModelV1Prompt = [
      { role: "user", content: "Hello" }
    ];

    const result = await model.doGenerate({
      prompt,
      maxTokens: 100,
      temperature: 0,
    });

    expect(result.text).toBe("Hello (generated)");
    const expectedSettings = {
      max_new_tokens: 100,
      do_sample: false, // because temperature === 0
      temperature: 0,
    };
    expect(result.rawCall.rawSettings).toEqual(expectedSettings);
  });

  it("should generate text with doGenerate using a message array prompt", async () => {
    const model = new HFTransformersjsChatLanguageModel(
        "test-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const promptMessages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ];

    const result = await model.doGenerate({
      prompt: promptMessages,
      maxTokens: 200,
      temperature: 0.5,
    });

    // The implementation formats messages appropriately
    expect(mockTextGenerationPipeline).toHaveBeenCalledWith(
        expect.stringContaining("Hello"),
        expect.objectContaining({
          max_new_tokens: 200,
          do_sample: true,
          temperature: 0.5
        })
    );
    expect(result.text).toBe("Hello (generated)");
  });

  it("should handle prompts with content as an array of parts", async () => {
    const model = new HFTransformersjsChatLanguageModel(
        "test-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const textPart1: LanguageModelV1TextPart = { text: "Hello" };
    const textPart2: LanguageModelV1TextPart = { text: "world" };

    const promptMessages = [
      { role: "user", content: [textPart1, textPart2] },
    ];

    const result = await model.doGenerate({
      prompt: promptMessages,
      maxTokens: 150,
      temperature: 0.7,
    });

    expect(mockTextGenerationPipeline).toHaveBeenCalledWith(
        "Hello world",
        expect.anything()
    );
    expect(result.text).toBe("Hello (generated)");
  });

  it("should stream tokens with doStream", async () => {
    const model = new HFTransformersjsChatLanguageModel(
        "test-model",
        {},
        { provider: "hf-test", apiKey: "dummy" }
    );

    const prompt: LanguageModelV1Prompt = [
      { role: "user", content: "Hello" }
    ];

    const { stream } = await model.doStream({
      prompt,
      maxTokens: 50,
      temperature: 0.7,
    } as LanguageModelV1CallOptions);

    // Test that the stream produces the expected tokens
    const reader = stream.getReader();
    const tokens = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      tokens.push(value);
    }

    expect(tokens).toEqual([
      { type: 'text-delta', textDelta: 'Hello' },
      { type: 'text-delta', textDelta: ' ' },
      { type: 'text-delta', textDelta: 'world' },
      { type: 'text-delta', textDelta: '!' }
    ]);

    expect(mockTextGenerationPipeline).toHaveBeenCalledWith(
        expect.stringContaining("Hello"),
        expect.objectContaining({
          max_new_tokens: 50,
          do_sample: true,
          temperature: 0.7,
          streamer: expect.any(TextStreamer)
        })
    );
  });
});