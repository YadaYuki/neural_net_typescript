import { Layer } from './base';
import nj from 'numjs';

export class Relu implements Layer {
  forward = (x: nj.NdArray<number>): nj.NdArray<number> => {
    const xArray = x.tolist();
    return nj.array(xArray.map((x) => (x > 0 ? x : 0)));
  };

  forwardBatch = (x: nj.NdArray<number[]>): nj.NdArray<number[]> => {
    const xArray = x.tolist();
    return nj.array(
      xArray.map((xArr: number[]) => {
        return xArr.map((x) => (x > 0 ? x : 0));
      })
    );
  };
  backward = (): void => {
    1 + 1;
  };
}
