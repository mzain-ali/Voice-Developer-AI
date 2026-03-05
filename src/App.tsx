/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from "@google/genai";
import { 
  Mic, MicOff, PhoneOff, MessageSquare, Globe, Sparkles, 
  Terminal, BookOpen, User, Code, UserCheck, History, 
  Settings, Play, X, ChevronRight, BarChart3, Briefcase, 
  Coffee, Target, Award, Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { floatToPcm16, base64ToBuffer, bufferToBase64, pcm16ToFloat } from './lib/audio-utils';

// Constants
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const MODEL = "gemini-2.5-flash-native-audio-preview-09-2025";

type AppMode = 'friendly' | 'interviewer';
type VoiceName = 'Zephyr' | 'Puck' | 'Charon' | 'Kore' | 'Fenrir';

interface Idiom {
  word: string;
  definition: string;
}

interface CodeSnippet {
  language: string;
  code: string;
}

interface SessionStats {
  mistakesCount: number;
  newWordsCount: number;
  fluencyScore: number;
  feedback: string;
}

const VOICES: { name: VoiceName; label: string; description: string }[] = [
  { name: 'Zephyr', label: 'Zephyr', description: 'Friendly & Energetic' },
  { name: 'Puck', label: 'Puck', description: 'Professional & Clear' },
  { name: 'Charon', label: 'Charon', description: 'Deep & Authoritative' },
  { name: 'Kore', label: 'Kore', description: 'Soft & Encouraging' },
  { name: 'Fenrir', label: 'Fenrir', description: 'Fast & Tech-focused' },
];

const SCENARIOS = [
  { id: 'standup', label: 'Daily Standup', icon: Coffee, prompt: 'Let\'s practice a Daily Standup. I am your Scrum Master. Tell me what you did yesterday, what you are doing today, and if you have any blockers.' },
  { id: 'review', label: 'Code Review', icon: Code, prompt: 'Let\'s practice a Code Review. I am a Senior Developer reviewing your Pull Request. I have some concerns about the complexity of your implementation. Defend your choices.' },
  { id: 'interview', label: 'System Design', icon: Target, prompt: 'Let\'s do a System Design interview. Design a scalable URL shortener like Bitly. I will be the interviewer asking about load balancing and database choices.' },
  { id: 'negotiation', label: 'Salary Negotiation', icon: Award, prompt: 'Let\'s practice a Salary Negotiation. You just received a job offer but the base salary is lower than you expected. Try to negotiate for a higher base or more equity.' },
];

const getSystemInstruction = (mode: AppMode) => `You are an AI voice conversation partner designed to help a developer improve their English communication skills.

Current Mode: ${mode === 'friendly' ? 'Friendly Partner' : 'Strict Technical Interviewer'}

Your role:
- ${mode === 'friendly' ? 'Act like a friendly conversation partner during a live voice call.' : 'Act as a strict, professional technical interviewer. Ask challenging questions and push for detailed technical answers.'}
- Speak naturally, clearly, and only in English.
- Help the user improve their spoken English.

Conversation rules:
1. Always communicate only in English.
2. If the user makes grammar mistakes:
   - First respond naturally so the conversation flows.
   - Then politely show the correction using the format:
     Corrected Sentence: <correct English sentence>
     Explanation: <short explanation of the mistake>

3. Focus conversations around developer topics: programming, software development, coding practices, system design, APIs, debugging, AI, tech career growth, technical interviews.

4. **Technical Idioms**: Whenever you use a common developer idiom (e.g., "under the hood", "technical debt", "boilerplate"), also include it at the end of your response in this format: [IDIOM: word | definition].

5. **Code Snippets**: If you explain a technical concept that benefits from code, include a snippet in this format: [CODE: language | code].

6. **Session Summary**: If the user says "End Session" or "Show Summary", provide a summary in this format:
   [SUMMARY]
   Mistakes: <number>
   New Words: <number>
   Fluency: <score 1-10>
   Feedback: <overall feedback on communication skills>

7. **Translation Mode**: If the user speaks in a language other than English, you MUST detect the language, translate it to English, and respond in English. You MUST use this EXACT format for your response:
   Translated Text: <the user's text translated to English>
   Response: <your natural response in English to that translated text>

8. Encourage the user to speak more. Ask follow-up questions.
9. Keep explanations short and clear for voice conversation.
10. Occasionally introduce useful English phrases for professional conversations.`;

interface TranscriptItem {
  id: string;
  role: 'user' | 'model';
  text: string;
  type?: 'correction' | 'translation' | 'normal';
}

export default function App() {
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [isMuted, setIsMuted] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  // New State for 6 Features
  const [mode, setMode] = useState<AppMode>('friendly');
  const [selectedVoice, setSelectedVoice] = useState<VoiceName>('Zephyr');
  const [idioms, setIdioms] = useState<Idiom[]>([]);
  const [codeSnippets, setCodeSnippets] = useState<CodeSnippet[]>([]);
  const [sessionSummary, setSessionSummary] = useState<SessionStats | null>(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const sessionRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioQueueRef = useRef<Float32Array[]>([]);
  const nextStartTimeRef = useRef(0);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // Persistence: Load from localStorage
  useEffect(() => {
    const savedIdioms = localStorage.getItem('devtalk_idioms');
    const savedSnippets = localStorage.getItem('devtalk_snippets');
    if (savedIdioms) setIdioms(JSON.parse(savedIdioms));
    if (savedSnippets) setCodeSnippets(JSON.parse(savedSnippets));
  }, []);

  // Persistence: Save to localStorage
  useEffect(() => {
    localStorage.setItem('devtalk_idioms', JSON.stringify(idioms));
  }, [idioms]);

  useEffect(() => {
    localStorage.setItem('devtalk_snippets', JSON.stringify(codeSnippets));
  }, [codeSnippets]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcript]);

  // Initialize Audio Context
  const initAudio = async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  };

  // Play audio from queue using scheduling for smoothness
  const playNextInQueue = useCallback(async () => {
    if (audioQueueRef.current.length === 0 || !audioContextRef.current) return;

    const chunk = audioQueueRef.current.shift()!;
    // Model output is 24kHz, let the browser resample to the context rate (16kHz)
    const buffer = audioContextRef.current.createBuffer(1, chunk.length, OUTPUT_SAMPLE_RATE);
    buffer.getChannelData(0).set(chunk);
    
    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    
    const now = audioContextRef.current.currentTime;
    if (nextStartTimeRef.current < now) {
      nextStartTimeRef.current = now + 0.05; // Small buffer to prevent gaps
    }
    
    source.start(nextStartTimeRef.current);
    nextStartTimeRef.current += buffer.duration;
    
    // If there's more in the queue, schedule it immediately
    if (audioQueueRef.current.length > 0) {
      playNextInQueue();
    }
  }, []);

  const handleMessage = useCallback((message: LiveServerMessage) => {
    // Handle audio output
    const base64Audio = message.serverContent?.modelTurn?.parts?.find(p => p.inlineData)?.inlineData?.data;
    if (base64Audio) {
      const floatData = pcm16ToFloat(base64ToBuffer(base64Audio));
      audioQueueRef.current.push(floatData);
      playNextInQueue();
    }

    // Handle interruption
    if (message.serverContent?.interrupted) {
      audioQueueRef.current = [];
      nextStartTimeRef.current = 0;
    }

    // Handle transcription from model
    if (message.serverContent?.modelTurn) {
      let text = message.serverContent.modelTurn.parts.map(p => p.text).filter(Boolean).join(' ');
      
      if (text) {
        // Parse Idioms: [IDIOM: word | definition]
        const idiomRegex = /\[IDIOM:\s*([^|]+)\|\s*([^\]]+)\]/g;
        let match;
        while ((match = idiomRegex.exec(text)) !== null) {
          const word = match[1].trim();
          const definition = match[2].trim();
          setIdioms(prev => {
            if (prev.some(i => i.word === word)) return prev;
            return [...prev, { word, definition }];
          });
        }
        text = text.replace(idiomRegex, '').trim();

        // Parse Code Snippets: [CODE: language | code]
        const codeRegex = /\[CODE:\s*([^|]+)\|\s*([^\]]+)\]/g;
        while ((match = codeRegex.exec(text)) !== null) {
          const language = match[1].trim();
          const code = match[2].trim();
          setCodeSnippets(prev => [...prev, { language, code }]);
        }
        text = text.replace(codeRegex, '').trim();

        // Parse Summary: [SUMMARY] Mistakes: X New Words: Y Fluency: Z Feedback: ...
        if (text.includes('[SUMMARY]')) {
          const summaryPart = text.split('[SUMMARY]')[1];
          const mistakes = parseInt(summaryPart.match(/Mistakes:\s*(\d+)/)?.[1] || '0');
          const newWords = parseInt(summaryPart.match(/New Words:\s*(\d+)/)?.[1] || '0');
          const fluency = parseInt(summaryPart.match(/Fluency:\s*(\d+)/)?.[1] || '0');
          const feedback = summaryPart.split('Feedback:')[1]?.trim() || '';
          
          setSessionSummary({
            mistakesCount: mistakes,
            newWordsCount: newWords,
            fluencyScore: fluency,
            feedback
          });
          text = text.split('[SUMMARY]')[0].trim();
        }

        if (text) {
          setTranscript(prev => [...prev, { id: Date.now().toString() + Math.random(), role: 'model', text }]);
        }
      }
    }
  }, [playNextInQueue]);

  const startCall = async (initialPrompt?: string) => {
    try {
      setIsConnecting(true);
      setSessionSummary(null); // Reset summary
      await initAudio();

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const sessionPromise = ai.live.connect({
        model: MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoice } },
          },
          systemInstruction: getSystemInstruction(mode),
        },
        callbacks: {
          onopen: () => {
            setIsConnecting(false);
            setIsActive(true);
            startMic();
            
            // If there's an initial prompt (from a scenario), send it
            if (initialPrompt && sessionRef.current) {
              sessionRef.current.sendRealtimeInput({
                text: initialPrompt
              });
            }
          },
          onmessage: (message) => {
            handleMessage(message);
          },
          onclose: () => {
            stopCall();
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            stopCall();
          }
        }
      });

      sessionRef.current = await sessionPromise;
    } catch (error) {
      console.error("Failed to start call:", error);
      setIsConnecting(false);
    }
  };


  const startMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const source = audioContextRef.current!.createMediaStreamSource(stream);
      const processor = audioContextRef.current!.createScriptProcessor(2048, 1, 1);
      processorRef.current = processor;

      const analyser = audioContextRef.current!.createAnalyser();
      analyser.fftSize = 256;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      source.connect(analyser);

      const updateAudioLevel = () => {
        if (!isActive) return;
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        setAudioLevel(average / 128);
        requestAnimationFrame(updateAudioLevel);
      };
      updateAudioLevel();

      processor.onaudioprocess = (e) => {
        if (isMuted || !sessionRef.current) return;
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Resample to 16000 if the context is running at a different rate
        let resampledData = inputData;
        const currentRate = audioContextRef.current!.sampleRate;
        if (currentRate !== 16000) {
          const ratio = currentRate / 16000;
          const newLength = Math.floor(inputData.length / ratio);
          const result = new Float32Array(newLength);
          for (let i = 0; i < newLength; i++) {
            result[i] = inputData[Math.floor(i * ratio)];
          }
          resampledData = result;
        }

        const pcmData = floatToPcm16(resampledData);
        const base64Data = bufferToBase64(pcmData);
        
        sessionRef.current.sendRealtimeInput({
          media: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
        });
      };

      source.connect(processor);
      processor.connect(audioContextRef.current!.destination);
    } catch (error) {
      console.error("Mic access denied:", error);
      stopCall();
    }
  };

  const stopCall = () => {
    setIsActive(false);
    setIsConnecting(false);
    
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    audioQueueRef.current = [];
    nextStartTimeRef.current = 0;
  };

  const toggleMute = () => setIsMuted(!isMuted);

  const downloadReport = () => {
    if (!sessionSummary) return;
    
    const content = `
DEV TALK - SESSION REPORT
Date: ${new Date().toLocaleDateString()}
-----------------------------------
SUMMARY
Mistakes Found: ${sessionSummary.mistakesCount}
New Words Learned: ${sessionSummary.newWordsCount}
Fluency Score: ${sessionSummary.fluencyScore}/10

FEEDBACK
${sessionSummary.feedback}

-----------------------------------
NEW IDIOMS CAPTURED
${idioms.map(i => `- ${i.word}: ${i.definition}`).join('\n')}

-----------------------------------
TRANSCRIPT
${transcript.map(t => `${t.role.toUpperCase()}: ${t.text}`).join('\n\n')}
    `.trim();

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `devtalk-report-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#0a0502] text-[#e0d8d0] font-sans selection:bg-[#ff4e00]/30 overflow-hidden flex flex-col">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_30%,#3a1510_0%,transparent_60%)] opacity-40" />
        <motion.div 
          animate={{ 
            opacity: isActive ? [0.4, 0.6, 0.4] : 0.2,
            scale: isActive ? [1, 1.1, 1] : 1
          }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-0 bg-[radial-gradient(circle_at_10%_80%,#ff4e00_0%,transparent_50%)] opacity-20 blur-[60px]" 
        />
      </div>

      {/* Header */}
      <header className="relative z-10 p-6 flex justify-between items-center border-b border-white/5 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-[#ff4e00] flex items-center justify-center shadow-[0_0_20px_rgba(255,78,0,0.4)]">
            <Terminal className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">DevTalk</h1>
            <p className="text-xs text-white/40 uppercase tracking-widest font-mono">English Partner</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setTranscript([])}
            className="text-[10px] uppercase tracking-widest font-bold text-white/40 hover:text-white/60 transition-colors"
          >
            Clear Chat
          </button>
          <button 
            onClick={() => setShowSidebar(!showSidebar)}
            className="relative p-2 rounded-full bg-white/5 border border-white/10 text-white/60 hover:text-white transition-colors"
          >
            <History className="w-5 h-5" />
            {(idioms.length > 0 || codeSnippets.length > 0) && (
              <span className="absolute top-0 right-0 w-2 h-2 bg-[#ff4e00] rounded-full" />
            )}
          </button>
          <button 
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-full bg-white/5 border border-white/10 text-white/60 hover:text-white transition-colors"
          >
            <Settings className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10">
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-500 animate-pulse' : 'bg-white/20'}`} />
            <span className="text-xs font-medium text-white/60">{isActive ? 'Live Session' : 'Standby'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6 max-w-7xl mx-auto w-full overflow-hidden">
        <div className="flex w-full h-full gap-6">
          {/* Left/Main Area */}
          <div className="flex-1 flex flex-col min-w-0">
            <AnimatePresence mode="wait">
              {sessionSummary ? (
                <motion.div 
                  key="summary"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-white/5 border border-white/10 rounded-3xl p-8 max-w-2xl mx-auto w-full space-y-8 backdrop-blur-xl"
                >
                  <div className="text-center space-y-2">
                    <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-4">
                      <BarChart3 className="w-8 h-8 text-emerald-500" />
                    </div>
                    <h2 className="text-3xl font-bold tracking-tight">Session Summary</h2>
                    <p className="text-white/40">Great job! Here is how you performed today.</p>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-white/5 p-4 rounded-2xl text-center border border-white/5">
                      <p className="text-2xl font-bold text-red-400">{sessionSummary.mistakesCount}</p>
                      <p className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Mistakes</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-2xl text-center border border-white/5">
                      <p className="text-2xl font-bold text-blue-400">{sessionSummary.newWordsCount}</p>
                      <p className="text-[10px] uppercase tracking-widest text-white/40 font-bold">New Words</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-2xl text-center border border-white/5">
                      <p className="text-2xl font-bold text-emerald-400">{sessionSummary.fluencyScore}/10</p>
                      <p className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Fluency</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-white/60 flex items-center gap-2">
                      <Award className="w-4 h-4" /> Feedback
                    </h3>
                    <p className="text-white/80 leading-relaxed italic bg-white/5 p-4 rounded-xl border border-white/5">
                      "{sessionSummary.feedback}"
                    </p>
                  </div>

                  <div className="flex gap-4">
                    <button 
                      onClick={() => setSessionSummary(null)}
                      className="flex-1 py-4 rounded-xl bg-white text-black font-bold hover:bg-white/90 transition-colors"
                    >
                      Start New Session
                    </button>
                    <button 
                      onClick={downloadReport}
                      className="px-6 py-4 rounded-xl bg-white/10 text-white font-bold hover:bg-white/20 transition-colors flex items-center gap-2"
                    >
                      <BarChart3 className="w-5 h-5" /> Export
                    </button>
                  </div>
                </motion.div>
              ) : !isActive && !isConnecting ? (
                <motion.div 
                  key="start"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="text-center space-y-8 py-12"
                >
                  <div className="space-y-4">
                    <h2 className="text-5xl md:text-7xl font-light tracking-tighter leading-none">
                      Level up your <br />
                      <span className="text-[#ff4e00] italic font-serif">English</span> for Devs
                    </h2>
                    <p className="text-white/60 max-w-md mx-auto text-lg leading-relaxed">
                      Real-time voice practice focused on programming, system design, and career growth.
                    </p>
                  </div>

                  <div className="flex flex-col items-center gap-6">
                    <button 
                      onClick={() => startCall()}
                      className="group relative px-12 py-5 rounded-full bg-white text-black font-semibold text-lg overflow-hidden transition-transform active:scale-95 hover:shadow-[0_0_40px_rgba(255,255,255,0.2)]"
                    >
                      <span className="relative z-10 flex items-center gap-2">
                        Start Conversation <Sparkles className="w-5 h-5" />
                      </span>
                      <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-cyan-400 opacity-0 group-hover:opacity-10 transition-opacity" />
                    </button>

                    <div className="w-full max-w-3xl space-y-4">
                      <h3 className="text-xs uppercase tracking-[0.2em] text-white/30 font-bold">Choose a Scenario</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {SCENARIOS.map((s) => (
                          <button
                            key={s.id}
                            onClick={() => startCall(s.prompt)}
                            className="flex flex-col items-center gap-3 p-4 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all group"
                          >
                            <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-[#ff4e00]/20 transition-colors">
                              <s.icon className="w-5 h-5 text-white/60 group-hover:text-[#ff4e00]" />
                            </div>
                            <span className="text-xs font-bold tracking-tight text-white/60 group-hover:text-white">{s.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div 
                  key="active"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="w-full h-full flex flex-col gap-8"
                >
                  {/* Visualizer Area */}
                  <div className="flex-1 flex flex-col items-center justify-center relative">
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <motion.div 
                        animate={{ 
                          scale: [1, 1.2 + audioLevel, 1],
                          opacity: [0.1, 0.3, 0.1]
                        }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="w-64 h-64 rounded-full bg-[#ff4e00] blur-[80px]"
                      />
                    </div>

                    <div className="relative flex items-center justify-center">
                      <motion.div 
                        animate={{ 
                          scale: isActive ? [1, 1.05 + audioLevel * 0.2, 1] : 1,
                          rotate: isActive ? [0, 5, -5, 0] : 0
                        }}
                        transition={{ duration: 3, repeat: Infinity }}
                        className="w-48 h-48 rounded-full bg-gradient-to-br from-[#ff4e00] to-[#3a1510] flex items-center justify-center shadow-[0_0_50px_rgba(255,78,0,0.3)] border border-white/10"
                      >
                        <div className="w-40 h-40 rounded-full bg-black/40 backdrop-blur-xl flex items-center justify-center overflow-hidden">
                          <div className="flex gap-1 items-end h-12">
                            {[...Array(8)].map((_, i) => (
                              <motion.div
                                key={i}
                                animate={{ 
                                  height: isActive ? [10, 10 + Math.random() * 40 * audioLevel, 10] : 4 
                                }}
                                transition={{ duration: 0.2, repeat: Infinity }}
                                className="w-1 bg-[#ff4e00] rounded-full"
                              />
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    <div className="mt-12 text-center space-y-2">
                      <h3 className="text-2xl font-medium tracking-tight">
                        {isConnecting ? 'Establishing connection...' : 'Listening...'}
                      </h3>
                      <p className="text-white/40 text-sm font-mono uppercase tracking-widest">
                        {isMuted ? 'Microphone Muted' : 'Speak Naturally'}
                      </p>
                    </div>
                  </div>

                  {/* Transcript Area */}
                  <div className="h-48 w-full max-w-2xl mx-auto overflow-y-auto px-4 space-y-4 mask-fade-edges scroll-smooth">
                    {transcript.length === 0 && !isConnecting && (
                      <div className="text-center text-white/20 italic py-8">
                        Start speaking to see the transcript...
                      </div>
                    )}
                    {transcript.map((item) => (
                      <motion.div 
                        key={item.id}
                        initial={{ opacity: 0, x: item.role === 'user' ? 10 : -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className={`flex gap-3 ${item.role === 'user' ? 'flex-row-reverse' : ''}`}
                      >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${item.role === 'user' ? 'bg-white/10' : 'bg-[#ff4e00]/20'}`}>
                          {item.role === 'user' ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4 text-[#ff4e00]" />}
                        </div>
                        <div className={`max-w-[80%] p-3 rounded-2xl text-sm leading-relaxed ${
                          item.role === 'user' 
                            ? 'bg-white/5 text-white/80 rounded-tr-none' 
                            : 'bg-[#ff4e00]/10 text-[#ff4e00] border border-[#ff4e00]/20 rounded-tl-none'
                        }`}>
                          {item.text.includes('Corrected Sentence:') ? (
                            <div className="space-y-3">
                              <p className="text-white/90 leading-relaxed">
                                {item.text.split('Corrected Sentence:')[0].trim()}
                              </p>
                              <div className="p-3 bg-black/40 rounded-xl border border-emerald-500/30 text-emerald-400 font-medium shadow-lg">
                                <div className="flex items-center gap-2 mb-2 opacity-60">
                                  <BookOpen className="w-3 h-3" />
                                  <span className="text-[10px] uppercase tracking-widest font-bold">Grammar Correction</span>
                                </div>
                                <p className="text-sm leading-relaxed">
                                  {item.text.split('Corrected Sentence:')[1].split('Explanation:')[0].trim()}
                                </p>
                              </div>
                              {item.text.includes('Explanation:') && (
                                <div className="p-3 bg-white/5 rounded-xl border border-white/10">
                                  <p className="text-xs text-white/60 leading-relaxed">
                                    <span className="font-bold text-white/40 uppercase text-[9px] tracking-wider block mb-1">Explanation</span>
                                    {item.text.split('Explanation:')[1].trim()}
                                  </p>
                                </div>
                              )}
                            </div>
                          ) : item.text.includes('Translated Text:') ? (
                            <div className="space-y-3">
                              <div className="p-3 bg-blue-500/10 rounded-xl border border-blue-500/30 text-blue-400 font-medium">
                                <div className="flex items-center gap-2 mb-2 opacity-60">
                                  <Globe className="w-3 h-3" />
                                  <span className="text-[10px] uppercase tracking-widest font-bold">Translation</span>
                                </div>
                                <p className="text-sm italic">
                                  {item.text.split('Translated Text:')[1].split('Response:')[0].trim()}
                                </p>
                              </div>
                              <p className="text-white/90 leading-relaxed">
                                {item.text.split('Response:')[1]?.trim() || item.text.split('Translated Text:')[0].trim()}
                              </p>
                            </div>
                          ) : (
                            item.text
                          )}
                        </div>
                      </motion.div>
                    ))}
                    <div ref={transcriptEndRef} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Sidebar Area */}
          <AnimatePresence>
            {showSidebar && (
              <motion.aside 
                initial={{ x: 400, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 400, opacity: 0 }}
                className="w-80 bg-white/5 border-l border-white/10 backdrop-blur-xl flex flex-col overflow-hidden rounded-l-3xl"
              >
                <div className="p-6 border-b border-white/10 flex justify-between items-center">
                  <h3 className="text-sm font-bold uppercase tracking-widest text-white/60">Knowledge Bank</h3>
                  <div className="flex items-center gap-2">
                    {(idioms.length > 0 || codeSnippets.length > 0) && (
                      <button 
                        onClick={() => {
                          if (confirm('Clear all saved idioms and snippets?')) {
                            setIdioms([]);
                            setCodeSnippets([]);
                            localStorage.removeItem('devtalk_idioms');
                            localStorage.removeItem('devtalk_snippets');
                          }
                        }}
                        className="text-[9px] uppercase tracking-widest font-bold text-red-400/60 hover:text-red-400 transition-colors px-2 py-1 rounded hover:bg-red-400/10"
                      >
                        Clear All
                      </button>
                    )}
                    <button onClick={() => setShowSidebar(false)} className="p-1 hover:bg-white/10 rounded-full transition-colors">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                
                <div className="flex-1 overflow-y-auto p-6 space-y-8">
                  {/* Idioms Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] uppercase tracking-[0.2em] text-[#ff4e00] font-bold flex items-center gap-2">
                      <Terminal className="w-3 h-3" /> Technical Idioms
                    </h4>
                    {idioms.length === 0 ? (
                      <p className="text-xs text-white/20 italic">No idioms captured yet.</p>
                    ) : (
                      <div className="space-y-3">
                        {idioms.map((i, idx) => (
                          <div key={idx} className="p-3 bg-white/5 rounded-xl border border-white/5 space-y-1">
                            <p className="text-sm font-bold text-white/90">{i.word}</p>
                            <p className="text-xs text-white/40 leading-relaxed">{i.definition}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Code Snippets Section */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] uppercase tracking-[0.2em] text-blue-400 font-bold flex items-center gap-2">
                      <Code className="w-3 h-3" /> Live Snippets
                    </h4>
                    {codeSnippets.length === 0 ? (
                      <p className="text-xs text-white/20 italic">No code explained yet.</p>
                    ) : (
                      <div className="space-y-3">
                        {codeSnippets.map((c, idx) => (
                          <div key={idx} className="p-3 bg-black/40 rounded-xl border border-white/5 space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-[9px] uppercase tracking-widest font-bold text-white/30">{c.language}</span>
                            </div>
                            <pre className="text-[10px] font-mono text-blue-300 overflow-x-auto p-2 bg-white/5 rounded-lg">
                              <code>{c.code}</code>
                            </pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </motion.aside>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowSettings(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="relative w-full max-w-lg bg-[#1a1a1a] border border-white/10 rounded-3xl overflow-hidden shadow-2xl"
            >
              <div className="p-6 border-b border-white/10 flex justify-between items-center">
                <h3 className="text-lg font-bold tracking-tight">Session Settings</h3>
                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-white/10 rounded-full transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-8 space-y-8">
                {/* Mode Selection */}
                <div className="space-y-4">
                  <label className="text-xs uppercase tracking-widest text-white/40 font-bold">Conversation Mode</label>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { id: 'friendly', label: 'Friendly Partner', icon: Coffee },
                      { id: 'interviewer', label: 'Tech Interviewer', icon: UserCheck }
                    ].map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setMode(m.id as AppMode)}
                        className={`flex items-center gap-3 p-4 rounded-2xl border transition-all ${
                          mode === m.id 
                            ? 'bg-[#ff4e00]/10 border-[#ff4e00] text-[#ff4e00]' 
                            : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10'
                        }`}
                      >
                        <m.icon className="w-5 h-5" />
                        <span className="font-bold text-sm">{m.label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Voice Selection */}
                <div className="space-y-4">
                  <label className="text-xs uppercase tracking-widest text-white/40 font-bold">AI Voice & Personality</label>
                  <div className="space-y-2">
                    {VOICES.map((v) => (
                      <button
                        key={v.name}
                        onClick={() => setSelectedVoice(v.name)}
                        className={`w-full flex items-center justify-between p-4 rounded-2xl border transition-all ${
                          selectedVoice === v.name 
                            ? 'bg-white/10 border-white/20 text-white' 
                            : 'bg-white/5 border-white/5 text-white/40 hover:bg-white/10'
                        }`}
                      >
                        <div className="text-left">
                          <p className="font-bold text-sm">{v.label}</p>
                          <p className="text-[10px] opacity-60">{v.description}</p>
                        </div>
                        {selectedVoice === v.name && <Play className="w-4 h-4 fill-current" />}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="p-6 bg-white/5 flex justify-end">
                <button 
                  onClick={() => setShowSettings(false)}
                  className="px-8 py-3 rounded-xl bg-white text-black font-bold hover:scale-105 active:scale-95 transition-all"
                >
                  Save Changes
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Footer Controls */}
      <footer className="relative z-10 p-8 flex justify-center items-center gap-6">
        {isActive && (
          <>
            <button 
              onClick={toggleMute}
              className={`w-14 h-14 rounded-full flex items-center justify-center transition-all active:scale-90 ${
                isMuted ? 'bg-red-500/20 text-red-500 border border-red-500/50' : 'bg-white/5 text-white hover:bg-white/10 border border-white/10'
              }`}
            >
              {isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
            </button>
            <button 
              onClick={stopCall}
              className="w-20 h-20 rounded-full bg-red-500 text-white flex items-center justify-center shadow-[0_0_30px_rgba(239,68,68,0.4)] hover:scale-105 active:scale-95 transition-all"
            >
              <PhoneOff className="w-8 h-8" />
            </button>
            <button 
              onClick={() => {
                if (sessionRef.current) {
                  sessionRef.current.sendRealtimeInput({ text: "End Session and show summary" });
                }
              }}
              className="px-6 h-14 rounded-full bg-white/5 text-white flex items-center gap-2 border border-white/10 hover:bg-white/10 transition-all font-bold text-xs uppercase tracking-widest"
            >
              <BarChart3 className="w-5 h-5" /> End Session
            </button>
          </>
        )}
      </footer>


      <style>{`
        .mask-fade-edges {
          mask-image: linear-gradient(to bottom, transparent 0%, black 10%, black 90%, transparent 100%);
        }
      `}</style>
    </div>
  );
}
