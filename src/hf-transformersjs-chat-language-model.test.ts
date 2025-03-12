import { describe, it, expect, vi, beforeEach } from "vitest";
import { HFTransformersjsChatLanguageModel } from "./hf-transformersjs-chat-language-model";
import { pipeline, TextStreamer } from "@huggingface/transformers";

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
  const fakePipelineResultNonStreaming = async (prompt: string, generationOptions: any) => {
    // For non-streaming we simply return a single result.
    return [{ generated_text: prompt + " (generated)" }];
  };

  const fakePipelineResultStreaming = async (prompt: string, generationOptions: any) => {
    // For streaming, our fake pipeline calls the provided callback one token at a time.
    // Assume tokens are: ["Hello", " ", "world", "!"]
    if (generationOptions.streamer && typeof generationOptions.streamer.callback_function === "function") {
      const tokens = ["Hello", " ", "world", "!"];
      for (const token of tokens) {
        generationOptions.streamer.callback_function(token);
      }
    }
    return [{ generated_text: "Hello world!" }];
  };

  beforeEach(() => {
    // Reset the mock before each test.
    (pipeline as any).mockReset();
  });

  it("should generate text with doGenerate using a string prompt", async () => {
    (pipeline as any).mockResolvedValue({
      tokenizer: { dummy: true },
      // When called, the fake pipeline returns a promise resolving an array
      // with generated_text: prompt + ' (generated)'
      call: fakePipelineResultNonStreaming,
    });
    // To simulate our usage we have our model call pipeline() internally.
    // Our HF class calls pn(prompt, generationOptions) so we mimic that
    (pipeline as any).mockImplementation(async () => fakePipelineResultNonStreaming);

    const model = new HFTransformersjsChatLanguageModel(
      "test-model",
      {},
      { provider: "hf-test", apiKey: "dummy" }
    );

    const result = await model.doGenerate({
      prompt: "Hello",
      maxTokens: 100,
      temperature: 0,
    });

    expect(result.text).toBe("Hello (generated)");
    const expectedBody = JSON.stringify({
      prompt: "Hello",
      max_new_tokens: 100,
      do_sample: false, // because temperature === 0
      temperature: 0,
    });
    expect(result.request.body).toBe(expectedBody);
  });

  it("should generate text with doGenerate using a message array prompt", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineResultNonStreaming);

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

    // The implementation joins the messages with newline.
    expect(result.text).toBe("Hello\nHi (generated)");
  });

  it("should stream tokens with doStream", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineResultStreaming);
    let collectedTokens: string[] = [];
    const onToken = (token: string) => {
      collectedTokens.push(token);
    };

    const model = new HFTransformersjsChatLanguageModel(
      "test-model",
      {},
      { provider: "hf-test", apiKey: "dummy" }
    );
    const result = await model.doStream({
      prompt: "Hello",
      maxTokens: 50,
      temperature: 0.7,
      onToken,
    });
    expect(collectedTokens).toEqual(["Hello", " ", "world", "!"]);
    // We check that the request body contains the correct prompt.
    expect(result.request.body).toContain('"prompt":"Hello"');
  });
});
