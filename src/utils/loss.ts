import nj from 'numjs';

export const crossEntroyError = (
  y: nj.NdArray<number>,
  t: nj.NdArray<number>
): number => {
  const logY = nj.array(y.tolist().map((yItem) => Math.log(yItem + 1e-7)));
  return -nj.sum(nj.multiply(t, logY));
};

export const crossEntroyErrorBatch = (
  yBatch: nj.NdArray<number[]>,
  tBatch: nj.NdArray<number[]>
): number => {
  const batchSize = tBatch.shape[0];
  const logYBatch = nj.array(
    yBatch.tolist().map((y) => y.map((yItem) => Math.log(yItem + 1e-7)))
  );
  return -nj.sum(nj.multiply(tBatch, logYBatch)) / batchSize;
};
