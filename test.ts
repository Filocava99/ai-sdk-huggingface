// index.ts
import {createHFTransformersjs} from "./src/hf-transformersjs-provider";
import {LanguageModelV1TextPart} from "@ai-sdk/provider";

async function main() {
    console.log("Starting the script...");
    const provider =  createHFTransformersjs({
        name: "hf-transformersjs",
    });
    console.log("Provider created");
    const llm = provider.languageModel("onnx-community/gemma-3-1b-it-ONNX", {
        provider: "hf-transformersjs",
    })
    console.log("Language model created");
    const userMessages:  Array<LanguageModelV1TextPart> = [
        {type: "text", text: "What is the capital of France?"},
        ]
    console.log("User messages created");
    const result = await llm.doGenerate({
        mode: {
            type: "regular"
        },
        inputFormat: "prompt",
        prompt: [{"role": "user", "content": userMessages}]
    })
    console.log(result)
}

// No while(true) loop - let the program exit naturally
main().catch(error => {
    console.error("[FATAL] Uncaught error:", error);
    process.exit(1);
});