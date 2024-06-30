// app/api/chat/route.ts
import { StreamingTextResponse } from "ai";
import { ChatMessage, MessageContent, TogetherLLM } from "llamaindex";
import { NextRequest, NextResponse } from "next/server";
import { createChatEngine } from "./engine";
import { LlamaIndexStream } from "./llamaindex-stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const convertMessageContent = (
  textMessage: string,
  imageUrl: string | undefined
): MessageContent => {
  if (!imageUrl) return textMessage;
  return [
    {
      type: "text",
      text: textMessage,
    },
    {
      type: "image_url",
      image_url: {
        url: imageUrl,
      },
    },
  ];
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { messages, data }: { messages: ChatMessage[]; data: any } = body;
    const userMessage = messages.pop();
    if (!messages || !userMessage || userMessage.role !== "user") {
      return NextResponse.json(
        {
          error:
            "messages are required in the request body and the last message must be from the user",
        },
        { status: 400 }
      );
    }

    const llm = new TogetherLLM({
      model: "meta-llama/Llama-3-8b-chat-hf",
      maxTokens: 1024,
      apiKey: process.env.TOGETHER_API_KEY,
    });

    const chatEngine = await createChatEngine(llm);

    const templatePrompt =
      "You are an assistant for question-answering tasks. Use the following pieces of retrieved context and provided data to answer the question. Absolutely do not answer questions that are not in the data. If you don't know the answer, just say that you don't know. Keep the answer concise and clear, always answer in Vietnamese, always say according to our data";

    const userMessageContent = convertMessageContent(
      userMessage.content + templatePrompt.toLowerCase(),
      data?.imageUrl
    );

    const response = await chatEngine.chat({
      message: userMessageContent,
      chatHistory: messages,
      stream: true,
    });

    const { stream, data: streamData } = LlamaIndexStream(response, {
      parserOptions: {
        image_url: data?.imageUrl,
      },
    });

    return new StreamingTextResponse(stream, {}, streamData);
  } catch (error) {
    console.error("[LlamaIndex]", error);
    return NextResponse.json(
      {
        error: (error as Error).message,
      },
      {
        status: 500,
      }
    );
  }
}
