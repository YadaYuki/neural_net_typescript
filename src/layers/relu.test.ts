import { Relu } from './relu';
import nj from 'numjs';

describe('Relu Layer Test', () => {
  describe('Relu.forward', () => {
    test('forward', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, 2, 3]);
      expect(relu.forward(x).tolist()).toEqual([0, 1, 2, 3]);
    });
    test('forward', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, -2, 0]);
      expect(relu.forward(x).tolist()).toEqual([0, 1, 0, 0]);
    });
  });
  describe('Relu.forwardBatch', () => {
    test('forwardBatch', () => {
      const relu = new Relu();
      const x = nj.array([
        [-1, 1, -2, 0],
        [-1, 1, -2, 0],
      ]);
      expect(relu.forwardBatch(x).tolist()).toEqual([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
      ]);
    });
  });
  describe('Relu.backward', () => {
    test('[-1, 10, -2, 0] backward() should return [0, 1, 0, 0]', () => {
      const relu = new Relu(nj.array([-1, 10, -2, 0]));
      expect(relu.backward().tolist()).toEqual([0, 1, 0, 0]);
    });
    test('[-1, -1, 3, 0] backward() should return [0, 0, 1, 0]', () => {
      const relu = new Relu(nj.array([-1, -1, 3, 0]));
      expect(relu.backward().tolist()).toEqual([0, 0, 1, 0]);
    });
  });
});
