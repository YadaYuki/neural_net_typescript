import { Layer } from './base';
import nj from 'numjs';
import { im2col, col2im } from '../utils/cnn';

export class Convolution implements Layer {
  W: nj.NdArray<number[][][]>;
  b: nj.NdArray<number>;
  stride: number;
  pad: number;
  dW: nj.NdArray<number[][][]>;
  db: nj.NdArray<number>;
  // for backward
  x: nj.NdArray<number[][][]>;
  colX: nj.NdArray<number[]>;
  colW: nj.NdArray<number[]>;
  constructor(
    W: nj.NdArray<number[][][]>,
    b: nj.NdArray<number>,
    stride = 1,
    pad = 0
  ) {
    this.W = W;
    this.b = b;
    this.stride = stride;
    this.pad = pad;
    this.dW = nj.zeros(0);
    this.db = nj.zeros(0);
    this.x = nj.zeros(0);
    this.colX = nj.zeros(0);
    this.colW = nj.zeros(0);
  }
  forward(): void {
    return;
  }
  forwardBatch(xBatch: nj.NdArray<number[][][]>): nj.NdArray<number[][][]> {
    const [FN, C, FH, FW] = this.W.shape;
    const [N, , H, W] = xBatch.shape;
    const outH = Math.floor(1 + (H + 2 * this.pad - FH) / this.stride);
    const outW = Math.floor(1 + (W + 2 * this.pad - FW) / this.stride);
    const colX = im2col(xBatch, FH, FW, this.stride, this.pad);
    const colW = (this.W.reshape(FN, FH * FW * C) as nj.NdArray<number[]>).T;
    const bMat = nj
      .dot(nj.ones([outH * outW * N, 1]), this.b.reshape(1, FN))
      .reshape(outH * outW * N, FN) as nj.NdArray<number[]>;
    const out = nj.add(nj.dot(colX, colW), bMat);

    this.x = xBatch;
    this.colX = colX;
    this.colW = colW;
    return (
      out.reshape(N, outH, outW, FN) as nj.NdArray<number[][][]>
    ).transpose(0, 3, 1, 2);
  }
  backward(): void {
    return;
  }
  backwardBatch(dout: nj.NdArray<number[][][]>): nj.NdArray<number[][][]> {
    const [FN, C, FH, FW] = this.W.shape;
    const [N, , H, W] = this.x.shape;
    const outH = Math.floor(1 + (H + 2 * this.pad - FH) / this.stride);
    const outW = Math.floor(1 + (W + 2 * this.pad - FW) / this.stride);
    const colDout = dout
      .transpose(0, 2, 3, 1)
      .reshape(outH * outW * N, FN) as nj.NdArray<number[]>;
    this.db = nj
      .dot(colDout.T, nj.ones(outH * outW * N).reshape(outH * outW * N, 1))
      .flatten();
    this.dW = nj
      .dot(this.colX.T, colDout)
      .transpose(1, 0)
      .reshape(FN, C, FH, FW);
    const colDx = nj.dot(colDout, this.colW.T);
    const [n, d, h, w] = this.x.shape;
    const dx = col2im(colDx, { n, d, h, w }, FH, FW, this.stride, this.pad);
    return dx;
  }
}
