import nj from 'numjs';

export const softmax = (x: nj.NdArray<number>): nj.NdArray<number> => {
  x = x.add(-x.max());
  return nj.divide(nj.exp(x), x.exp().sum());
};

export const softmaxBatch = (
  xBatch: nj.NdArray<number[]>
): nj.NdArray<number[]> => {
  return nj.array(xBatch.tolist().map((x) => softmax(nj.array(x)).tolist()));
};
