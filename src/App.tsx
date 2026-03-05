/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from "@google/genai";
import { Mic, MicOff, PhoneOff, MessageSquare, Globe, Sparkles, Terminal, BookOpen, User } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { floatToPcm16, base64ToBuffer, bufferToBase64, pcm16ToFloat } from './lib/audio-utils';

// Constants
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const MODEL = "gemini-2.5-flash-native-audio-preview-09-2025";

const SYSTEM_INSTRUCTION = `You are an AI voice conversation partner designed to help a developer improve their English communication skills.

Your role:
- Act like a friendly conversation partner during a live voice call.
- Speak naturally, clearly, and only in English.
- Help the user improve their spoken English.

Conversation rules:

1. Always communicate only in English.
If the user speaks in another language, translate it into English and respond in English.

2. If the user makes grammar mistakes:
- First respond naturally so the conversation flows.
- Then politely show the correction.

Use this format:
Corrected Sentence: <correct English sentence>
Explanation: <short explanation of the mistake>

3. Focus conversations around developer topics such as:
- programming
- software development
- coding practices
- system design
- APIs
- debugging
- AI and technology
- developer career growth
- technical interviews

4. Maintain memory of the active voice conversation.
Use previous messages in the conversation to maintain context just like a real phone call.

5. Encourage the user to speak more. Ask follow-up questions related to software development.

6. Keep explanations short and clear because this is a voice conversation.

7. If the user switches to translation mode:
- Listen to the user’s speech
- Detect the language
- Translate it to English
- Show the translated text
- Respond only in English

8. Be supportive and encouraging. Never criticize the user harshly. Always guide them constructively.

9. Keep responses concise and conversational because the user is speaking in real time.

10. Occasionally introduce useful English phrases that developers commonly use in professional conversations.`;

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

  const audioContextRef = useRef<AudioContext | null>(null);
  const sessionRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioQueueRef = useRef<Float32Array[]>([]);
  const nextStartTimeRef = useRef(0);

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
      const text = message.serverContent.modelTurn.parts.map(p => p.text).filter(Boolean).join(' ');
      if (text) {
        setTranscript(prev => {
          // Avoid duplicates if the same text comes in multiple chunks (though usually chunks are unique)
          // For Live API, chunks are usually parts of the same turn.
          // We'll just append for now, but in a real app we might want to group by turn.
          return [...prev, { id: Date.now().toString() + Math.random(), role: 'model', text }];
        });
      }
    }
  }, [playNextInQueue]);

  const startCall = async () => {
    try {
      setIsConnecting(true);
      await initAudio();

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const sessionPromise = ai.live.connect({
        model: MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction: SYSTEM_INSTRUCTION,
        },
        callbacks: {
          onopen: () => {
            setIsConnecting(false);
            setIsActive(true);
            startMic();
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
            Clear History
          </button>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10">
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-500 animate-pulse' : 'bg-white/20'}`} />
            <span className="text-xs font-medium text-white/60">{isActive ? 'Live Session' : 'Standby'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6 max-w-5xl mx-auto w-full">
        <AnimatePresence mode="wait">
          {!isActive && !isConnecting ? (
            <motion.div 
              key="start"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="text-center space-y-8"
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

              <button 
                onClick={startCall}
                className="group relative px-12 py-5 rounded-full bg-white text-black font-semibold text-lg overflow-hidden transition-transform active:scale-95 hover:shadow-[0_0_40px_rgba(255,255,255,0.2)]"
              >
                <span className="relative z-10 flex items-center gap-2">
                  Start Conversation <Sparkles className="w-5 h-5" />
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-cyan-400 opacity-0 group-hover:opacity-10 transition-opacity" />
              </button>

              <div className="grid grid-cols-3 gap-4 pt-12">
                {[
                  { icon: Terminal, label: "Tech Topics" },
                  { icon: BookOpen, label: "Grammar Help" },
                  { icon: Globe, label: "Translation" }
                ].map((item, i) => (
                  <div key={i} className="flex flex-col items-center gap-2 text-white/40">
                    <item.icon className="w-5 h-5" />
                    <span className="text-[10px] uppercase tracking-widest font-bold">{item.label}</span>
                  </div>
                ))}
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
              <div className="h-48 w-full max-w-2xl mx-auto overflow-y-auto px-4 space-y-4 mask-fade-edges">
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
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

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
            <button className="w-14 h-14 rounded-full bg-white/5 text-white flex items-center justify-center border border-white/10 hover:bg-white/10 transition-all">
              <MessageSquare className="w-6 h-6" />
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
