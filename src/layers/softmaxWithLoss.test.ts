import { SoftmaxWithLoss } from './softmaxWithLoss';
import nj from 'numjs';

describe('SoftmaxWithLoss Layer Test', () => {
  describe('SoftmaxWithLoss.backward', () => {
    // forwardに関しては,utils以下のロジックがほとんどであるため、ここでは省略。
    test('backward', () => {
      const y = nj.array([1, 2, 3, 4]);
      const t = nj.array([0, 1, 2, 3]);
      const softmaxWithLoss = new SoftmaxWithLoss(y, undefined, t);
      expect(softmaxWithLoss.backward().tolist()).toEqual([1, 1, 1, 1]);
    });
    test('backward', () => {
      const y = nj.array([0, 1, 3, 1]);
      const t = nj.array([11, 2, 4, 5]);
      const softmaxWithLoss = new SoftmaxWithLoss(y, undefined, t);
      expect(softmaxWithLoss.backward().tolist()).toEqual([-11, -1, -1, -4]);
    });
  });
  describe('SoftmaxWithLoss.backwardBatch', () => {
    // forwardに関しては,utils以下のロジックがほとんどであるため、ここでは省略。
    test('backward', () => {
      const y = nj.array([1, 2, 3, 4]);
      const t = nj.array([0, 1, 2, 3]);
      const softmaxWithLoss = new SoftmaxWithLoss(y, undefined, t);
      expect(softmaxWithLoss.backward().tolist()).toEqual([1, 1, 1, 1]);
    });
    test('backward', () => {
      const yBatch = nj.array([
        [1, 2, 3, 4],
        [0, 1, 3, 1],
      ]);
      const tBatch = nj.array([
        [0, 1, 2, 3],
        [11, 2, 4, 5],
      ]);
      const softmaxWithLoss = new SoftmaxWithLoss(
        undefined,
        yBatch,
        undefined,
        tBatch
      );
      expect(softmaxWithLoss.backwardBatch().tolist()).toEqual([
        [0.5, 0.5, 0.5, 0.5],
        [-5.5, -0.5, -0.5, -2],
      ]);
    });
  });
});
