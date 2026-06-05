import {
  Loader2,
  Music2,
  Paperclip,
  Plus,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Wrench
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Engine = "auto" | "openai" | "keyword";

type SessionInfo = {
  id: string;
  requested_engine?: string;
  engine?: string;
  fallback_reason?: string | null;
};

type Artifact = {
  id: string;
  name: string;
  kind: string;
  mime_type: string;
  size_bytes: number;
  url: string;
};

type AgentEvent = {
  type: string;
  text?: string;
  message?: string;
  tool?: string;
  ok?: boolean;
  artifact?: Artifact;
  data?: Record<string, unknown>;
  error?: string;
  [key: string]: unknown;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  events: AgentEvent[];
  artifacts: Artifact[];
  pending?: boolean;
};

const SESSION_KEY = "music-agent-web-session";

export function App() {
  const [session, setSession] = useState<SessionInfo | null>(() => {
    const saved = localStorage.getItem(SESSION_KEY);
    return saved ? { id: saved, engine: "restored" } : null;
  });
  const [settings, setSettings] = useState({
    agent_engine: "auto" as Engine,
    openai_model: "",
    max_steps: 8
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [audioArtifact, setAudioArtifact] = useState<Artifact | null>(null);
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!session) {
      void createSession({ resetMessages: false });
    }
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const engineLabel = useMemo(() => {
    if (!session) {
      return "starting";
    }
    return session.engine || session.requested_engine || "ready";
  }, [session]);

  async function createSession(options: { resetMessages: boolean }): Promise<SessionInfo> {
    setError(null);
    const response = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_engine: settings.agent_engine,
        openai_model: settings.openai_model || undefined,
        max_steps: settings.max_steps
      })
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || payload.error || "Could not create session.");
    }
    const nextSession = payload.session as SessionInfo;
    localStorage.setItem(SESSION_KEY, nextSession.id);
    setSession(nextSession);
    if (options.resetMessages) {
      setMessages([]);
    }
    return nextSession;
  }

  async function ensureSession(): Promise<SessionInfo> {
    if (session?.id) {
      return session;
    }
    return createSession({ resetMessages: false });
  }

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    const content = input.trim();
    if (!content || sending) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      events: [],
      artifacts: []
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      events: [],
      artifacts: [],
      pending: true
    };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setSending(true);
    setError(null);

    try {
      const currentSession = await ensureSession();
      let response = await postMessageStream(currentSession.id, content);
      if (!response.ok && response.status === 400) {
        const text = await response.text();
        if (text.includes("Unknown session id")) {
          const freshSession = await createSession({ resetMessages: false });
          response = await postMessageStream(freshSession.id, content);
        } else {
          throw new Error(text);
        }
      }
      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }
      await readSse(response, (agentEvent) => applyAgentEvent(assistantId, agentEvent));
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? { ...item, content: message, pending: false, events: [...item.events, { type: "error", error: message }] }
            : item
        )
      );
    } finally {
      setSending(false);
      setMessages((current) => current.map((item) => (item.id === assistantId ? { ...item, pending: false } : item)));
    }
  }

  async function postMessageStream(sessionId: string, content: string) {
    return fetch(`/api/sessions/${sessionId}/messages/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content,
        audio_artifact_id: audioArtifact?.id
      })
    });
  }

  function applyAgentEvent(messageId: string, agentEvent: AgentEvent) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) {
          return message;
        }
        if (agentEvent.type === "assistant_delta") {
          return { ...message, content: message.content + (agentEvent.text || "") };
        }
        if (agentEvent.type === "artifact" && agentEvent.artifact) {
          return { ...message, artifacts: mergeArtifacts(message.artifacts, [agentEvent.artifact]) };
        }
        if (agentEvent.type === "final") {
          const data = agentEvent.data || {};
          const finalArtifacts = Array.isArray(data.artifacts) ? (data.artifacts as Artifact[]) : [];
          const fallback = message.content || finalText(data);
          return {
            ...message,
            content: fallback,
            artifacts: mergeArtifacts(message.artifacts, finalArtifacts),
            events: [...message.events, agentEvent],
            pending: false
          };
        }
        if (agentEvent.type === "error") {
          return {
            ...message,
            content: agentEvent.error || "Request failed.",
            events: [...message.events, agentEvent],
            pending: false
          };
        }
        return { ...message, events: [...message.events, agentEvent] };
      })
    );
  }

  async function uploadAudio(file: File | undefined) {
    if (!file) {
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const currentSession = await ensureSession();
      const form = new FormData();
      form.append("file", file);
      form.append("session_id", currentSession.id);
      const response = await fetch("/api/uploads/audio", {
        method: "POST",
        body: form
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.detail || payload.error || "Upload failed.");
      }
      setAudioArtifact(payload.artifact as Artifact);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function clearSession() {
    if (!session?.id) {
      return;
    }
    setError(null);
    await fetch(`/api/sessions/${session.id}/clear`, { method: "POST" });
    setMessages([]);
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <Music2 size={22} />
          </div>
          <div>
            <h1>Music Agent</h1>
            <span>{engineLabel}</span>
          </div>
        </div>

        <section className="panel">
          <header>
            <SlidersHorizontal size={16} />
            <span>会话</span>
          </header>
          <label>
            <span>Engine</span>
            <select
              value={settings.agent_engine}
              onChange={(event) => setSettings({ ...settings, agent_engine: event.target.value as Engine })}
            >
              <option value="auto">auto</option>
              <option value="openai">openai</option>
              <option value="keyword">keyword</option>
            </select>
          </label>
          <label>
            <span>Model</span>
            <input
              value={settings.openai_model}
              onChange={(event) => setSettings({ ...settings, openai_model: event.target.value })}
              placeholder="默认模型"
            />
          </label>
          <label>
            <span>Steps</span>
            <input
              type="number"
              min={1}
              max={32}
              value={settings.max_steps}
              onChange={(event) => setSettings({ ...settings, max_steps: Number(event.target.value) })}
            />
          </label>
          <div className="buttonRow">
            <button type="button" onClick={() => void createSession({ resetMessages: true })}>
              <Plus size={16} />
              新会话
            </button>
            <button type="button" onClick={() => void clearSession()}>
              <RotateCcw size={16} />
              清空
            </button>
          </div>
        </section>

        <section className="panel">
          <header>
            <Paperclip size={16} />
            <span>音频</span>
          </header>
          <button type="button" className="uploadButton" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            {uploading ? <Loader2 className="spin" size={16} /> : <Paperclip size={16} />}
            上传音频
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.ncm"
            hidden
            onChange={(event) => void uploadAudio(event.target.files?.[0])}
          />
          {audioArtifact && (
            <div className="audioChip">
              <span>{audioArtifact.name}</span>
              <button type="button" title="移除音频" onClick={() => setAudioArtifact(null)}>
                ×
              </button>
            </div>
          )}
        </section>

        {session?.fallback_reason && <p className="note">{session.fallback_reason}</p>}
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="emptyState">
              <Music2 size={32} />
              <p>今天想处理哪段音乐？</p>
            </div>
          )}
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="bubble">
                {message.pending && !message.content ? (
                  <span className="thinking">
                    <Loader2 className="spin" size={16} />
                    thinking
                  </span>
                ) : (
                  <p>{message.content}</p>
                )}
                {message.events.length > 0 && <EventTimeline events={message.events} />}
                {message.artifacts.length > 0 && <ArtifactList artifacts={message.artifacts} />}
              </div>
            </article>
          ))}
          <div ref={scrollRef} />
        </div>

        {error && <div className="errorBar">{error}</div>}

        <form className="composer" onSubmit={submitMessage}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="输入你的音乐任务"
            rows={1}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submitMessage();
              }
            }}
          />
          <button type="submit" className="sendButton" title="发送" disabled={sending || !input.trim()}>
            {sending ? <Loader2 className="spin" size={19} /> : <Send size={19} />}
          </button>
        </form>
      </main>
    </div>
  );
}

function EventTimeline({ events }: { events: AgentEvent[] }) {
  const visible = events.filter((event) => event.type !== "final");
  if (visible.length === 0) {
    return null;
  }
  return (
    <details className="timeline">
      <summary>
        <Wrench size={15} />
        工具调用
      </summary>
      <ol>
        {visible.map((event, index) => (
          <li key={`${event.type}-${index}`}>
            <span>{eventSummary(event)}</span>
            <code>{event.type}</code>
          </li>
        ))}
      </ol>
    </details>
  );
}

function ArtifactList({ artifacts }: { artifacts: Artifact[] }) {
  return (
    <div className="artifacts">
      {artifacts.map((artifact) => (
        <div key={artifact.id} className="artifact">
          <div>
            <strong>{artifact.name}</strong>
            <span>{formatBytes(artifact.size_bytes)}</span>
          </div>
          {artifact.kind === "audio" && <audio controls src={artifact.url} />}
          <a href={artifact.url} download>
            下载
          </a>
        </div>
      ))}
    </div>
  );
}

async function readSse(response: Response, onEvent: (event: AgentEvent) => void) {
  const reader = response.body?.getReader();
  if (!reader) {
    return;
  }
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseBlock(block);
      if (parsed) {
        onEvent(parsed);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseSseBlock(block: string): AgentEvent | null {
  const dataLines = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return JSON.parse(dataLines.join("\n")) as AgentEvent;
  } catch {
    return null;
  }
}

function mergeArtifacts(existing: Artifact[], incoming: Artifact[]) {
  const byId = new Map(existing.map((artifact) => [artifact.id, artifact]));
  for (const artifact of incoming) {
    byId.set(artifact.id, artifact);
  }
  return Array.from(byId.values());
}

function finalText(data: Record<string, unknown>) {
  if (typeof data.final_answer === "string" && data.final_answer) {
    return data.final_answer;
  }
  if (typeof data.routed_to === "string") {
    return `已完成：${data.routed_to}`;
  }
  return "已完成。";
}

function eventSummary(event: AgentEvent) {
  if (event.type === "agent_step") {
    return String(event.message || "Agent step");
  }
  if (event.type === "tool_call") {
    return `调用 ${event.tool || "tool"}`;
  }
  if (event.type === "tool_result") {
    return `${event.tool || "tool"} ${event.ok ? "完成" : "失败"}`;
  }
  if (event.type === "error") {
    return String(event.error || "Error");
  }
  return event.type;
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
