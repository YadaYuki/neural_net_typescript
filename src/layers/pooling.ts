import { Layer } from './base';
import nj from 'numjs';
import { im2col } from '../utils/cnn';

export class Pooling implements Layer {
  poolH: number;
  poolW: number;
  stride: number;
  pad: number;

  constructor(poolH: number, poolW: number, stride = 1, pad = 0) {
    this.poolH = poolH;
    this.poolW = poolW;
    this.stride = stride;
    this.pad = pad;
  }

  forward(): void {
    return;
  }
  forwardBatch(xBatch: nj.NdArray<number[][][]>): nj.NdArray<number[][][]> {
    const [N, C, H, W] = xBatch.shape;
    const outH = Math.floor(1 + (H - this.poolH) / this.stride);
    const outW = Math.floor(1 + (W - this.poolW) / this.stride);
    const col = im2col(
      xBatch,
      this.poolH,
      this.poolW,
      this.stride,
      this.pad
    ).reshape(outH * outW * N * C, this.poolH * this.poolW) as nj.NdArray<
      number[]
    >;
    const out = (
      nj
        .array(
          col.tolist().map((convoluteOutput) => {
            return Math.max(...convoluteOutput);
          })
        )
        .reshape(N, outH, outW, C) as nj.NdArray<number[][][]>
    ).transpose(0, 3, 1, 2);
    return out;
  }
  backward(): void {
    return;
  }
  backwardBatch(): void {
    return;
  }
}
