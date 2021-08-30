import { crossEntroyError, crossEntroyErrorBatch } from './loss';
import nj from 'numjs';

describe('loss function test', () => {
  describe('crossEntropyError', () => {
    test('crossEntropyError([0,0,0.8,0.2],[0,0,1,0]) should be 0.2231434', () => {
      expect(
        crossEntroyError(nj.array([0, 0, 0.8, 0.2]), nj.array([0, 0, 1, 0]))
      ).toBeCloseTo(0.2231434, 7);
    });
    test('crossEntropyError([0.1,0.6,0.3,0.0],[0,1,0,0]) should be 0.5108255', () => {
      expect(
        crossEntroyError(nj.array([0.1, 0.6, 0.3, 0.0]), nj.array([0, 1, 0, 0]))
      ).toBeCloseTo(0.5108255, 7);
    });
  });
  describe('crossEntropyErrorBatch', () => {
    test('crossEntropyError([[0.1, 0.6, 0.3, 0.0], [0, 0, 0.8, 0.2]],[[0, 1, 0, 0], [0, 0, 1, 0]]) should be 0.3669844', () => {
      expect(
        crossEntroyErrorBatch(
          nj.array([
            [0.1, 0.6, 0.3, 0.0],
            [0, 0, 0.8, 0.2],
          ]),
          nj.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
          ])
        )
      ).toBeCloseTo(0.3669844, 7);
    });
    test('crossEntropyError([[0, 0.6, 0.4, 0.0], [0, 0.1, 0.7, 0.2]],[[0, 1, 0, 0], [0, 0, 1, 0]]) should be 0.4337501', () => {
      expect(
        crossEntroyErrorBatch(
          nj.array([
            [0, 0.6, 0.4, 0.0],
            [0, 0.1, 0.7, 0.2],
          ]),
          nj.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
          ])
        )
      ).toBeCloseTo(0.4337501, 7);
    });
  });
});
