/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export function floatToPcm16(float32Array: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

export function pcm16ToFloat(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  const float32Array = new Float32Array(buffer.byteLength / 2);
  for (let i = 0; i < float32Array.length; i++) {
    const s = view.getInt16(i * 2, true);
    float32Array[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
  }
  return float32Array;
}

export function base64ToBuffer(base64: string): ArrayBuffer {
  const binaryString = window.atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

export function bufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}
