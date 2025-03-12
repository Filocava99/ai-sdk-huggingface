import { describe, it, expect, vi, beforeEach } from "vitest";
import { HFTransformersjsCompletionLanguageModel } from "./hf-transformersjs-completion-language-model";
import { pipeline, TextStreamer } from "@huggingface/transformers";

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

describe("HFTransformersjsCompletionLanguageModel", () => {
  const fakePipelineCompletion = async (prompt: string, generationOptions: any) => {
    // Simply echo the prompt plus an appended generated text.
    return [{ generated_text: prompt + " (completed)" }];
  };

  const fakePipelineStreaming = async (prompt: string, generationOptions: any) => {
    // Simulate streaming tokens for completions.
    if (generationOptions.streamer && typeof generationOptions.streamer.callback_function === "function") {
      const tokens = ["This ", "is ", "completed."];
      for (const token of tokens) {
        generationOptions.streamer.callback_function(token);
      }
    }
    return [{ generated_text: "This is completed." }];
  };

  beforeEach(() => {
    (pipeline as any).mockReset();
  });

  it("should generate text with doGenerate using a plain text prompt", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineCompletion);

    const model = new HFTransformersjsCompletionLanguageModel(
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
    const expectedPayload = {
      prompt: "Test prompt",
      max_new_tokens: 150,
      do_sample: true, // temperature (0.3) > 0 so true
      temperature: 0.3,
    };
    expect(result.request.body).toBe(JSON.stringify(expectedPayload));
  });

  it("should stream tokens with doStream", async () => {
    (pipeline as any).mockImplementation(async () => fakePipelineStreaming);
    let tokensReceived: string[] = [];
    const onToken = (token: string) => tokensReceived.push(token);
    const model = new HFTransformersjsCompletionLanguageModel(
      "completion-model",
      {},
      { provider: "hf-test", apiKey: "dummy" }
    );
    const result = await model.doStream({
      prompt: "Test prompt",
      maxTokens: 150,
      temperature: 0.3,
      onToken,
    });
    expect(tokensReceived).toEqual(["This ", "is ", "completed."]);
    expect(result.request.body).toContain('"prompt":"Test prompt"');
  });
});
