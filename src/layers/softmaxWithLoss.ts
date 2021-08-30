import { Layer } from './base';
import nj from 'numjs';
import { softmax, softmaxBatch } from '../utils/activation';
import { crossEntroyError, crossEntroyErrorBatch } from '../utils/loss';

export class SoftmaxWithLoss implements Layer {
  y: nj.NdArray<number>;
  yBatch: nj.NdArray<number[]>;
  t: nj.NdArray<number>;
  tBatch: nj.NdArray<number[]>;
  loss: number;
  constructor(
    y: nj.NdArray<number> = nj.zeros(0),
    yBatch: nj.NdArray<number[]> = nj.zeros(0),
    t: nj.NdArray<number> = nj.zeros(0),
    tBatch: nj.NdArray<number[]> = nj.zeros(0)
  ) {
    this.y = y;
    this.yBatch = yBatch;
    this.t = t;
    this.tBatch = tBatch;
    this.loss = -1;
  }
  forward(x: nj.NdArray<number>, t: nj.NdArray<number>): number {
    this.y = softmax(x);
    this.t = t;
    this.loss = crossEntroyError(this.y, this.t);
    return this.loss;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  backward(dout = 1): nj.NdArray<number> {
    return this.y.subtract(this.t);
  }
  forwardBatch(
    xBatch: nj.NdArray<number[]>,
    tBatch: nj.NdArray<number[]>
  ): number {
    this.yBatch = softmaxBatch(xBatch);
    this.tBatch = tBatch;
    this.loss = crossEntroyErrorBatch(this.yBatch, this.tBatch);
    return this.loss;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  backwardBatch(dout = 1): nj.NdArray<number[]> {
    const batchSize = this.tBatch.shape[0];
    return this.yBatch.subtract(this.tBatch).divide(batchSize);
  }
}
