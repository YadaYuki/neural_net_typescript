import nj from 'numjs';
import { Layer } from './layers/base';
import { Affine } from './layers/affine';
import { Relu } from './layers/relu';
import { SoftmaxWithLoss } from './layers/softmaxWithLoss';

export class TwoLayerNet {
  W1: nj.NdArray<number[]>;
  b1: nj.NdArray<number>;
  W2: nj.NdArray<number[]>;
  b2: nj.NdArray<number>;
  layers: Layer[];
  lossLayer: Layer;
  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    this.W1 = nj
      .random([inputSize * hiddenSize])
      .multiply(0.01)
      .reshape(inputSize, hiddenSize) as nj.NdArray<number[]>;
    this.b1 = nj.zeros([hiddenSize]);
    this.W2 = nj
      .random([hiddenSize * outputSize])
      .multiply(0.01)
      .reshape(hiddenSize, outputSize) as nj.NdArray<number[]>;
    this.b2 = nj.zeros([outputSize]);
    this.layers = [
      new Affine(this.W1, this.b1),
      new Relu(),
      new Affine(this.W2, this.b2),
    ];
    this.lossLayer = new SoftmaxWithLoss();
  }

  /* 基本的にはバッチ学習であることを前提とする。 */
  forward(xBatch: nj.NdArray<number[]>, tBatch: nj.NdArray<number[]>): number {
    let scoreBatch: nj.NdArray<number[] | number> = xBatch;
    for (const layer of this.layers) {
      scoreBatch = layer.forwardBatch(scoreBatch);
    }
    const loss = this.lossLayer.forward(scoreBatch, tBatch);
    return loss;
  }
}