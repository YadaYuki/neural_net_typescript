import { softmax, softmaxBatch } from './activation';
import nj from 'numjs';

describe('activation test', () => {
  describe('softmax', () => {
    test('softmax([-2,0,-1,1]) should return [0.0320586  0.23688282 0.08714432 0.64391426] ', () => {
      const y = softmax(nj.array([-2, 0, -1, 1])).tolist();
      const expected = [0.0320586, 0.23688282, 0.08714432, 0.64391426];
      y.map((_, idx) => {
        expect(y[idx]).toBeCloseTo(expected[idx], 7);
      });
    });
    test('softmax([10000,10000,10000,10000]) should return [0.25,0.25,0,.25,0.25] ', () => {
      const y = softmax(nj.array([10000, 10000, 10000, 10000])).tolist();
      const expected = [0.25, 0.25, 0.25, 0.25];
      y.map((_, idx) => {
        expect(y[idx]).toBeCloseTo(expected[idx], 7);
      });
    });
  });
  describe('softmaxBatch', () => {
    test('softmaxBatch([[-2,0,-1,1],[10000,10000,10000,10000]]) should return [[0.0320586, 0.23688282, 0.08714432, 0.64391426],[0.25, 0.25, 0.25, 0.25]]', () => {
      const yBatch = softmaxBatch(
        nj.array([
          [-2, 0, -1, 1],
          [10000, 10000, 10000, 10000],
        ])
      ).tolist();
      const expected = [
        [0.0320586, 0.23688282, 0.08714432, 0.64391426],
        [0.25, 0.25, 0.25, 0.25],
      ];
      yBatch.map((y, i) => {
        y.map((_, j) => {
          expect(y[j]).toBeCloseTo(expected[i][j], 7);
        });
      });
    });
  });
});
