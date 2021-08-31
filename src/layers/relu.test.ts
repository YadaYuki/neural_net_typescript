import { Relu } from './relu';
import nj from 'numjs';

describe('Relu Layer Test', () => {
  describe('Relu.forward', () => {
    test('forward', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, 2, 3]);
      expect(relu.forward(x).tolist()).toEqual([-0, 1, 2, 3]);
    });
    test('forward', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, -2, 0]);
      expect(relu.forward(x).tolist()).toEqual([-0, 1, -0, 0]);
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
        [-0, 1, -0, 0],
        [-0, 1, -0, 0],
      ]);
    });
  });
  describe('Relu.backward', () => {
    test('x=[10,21,0,-1],dout = [-1,10,10,0] backward() should return [-1, 10, 0, 0]', () => {
      const relu = new Relu(nj.array([1, 1, 0, -0]));
      expect(relu.backward(nj.array([-1, 10, 10, 0])).tolist()).toEqual([
        -1, 10, 0, -0,
      ]);
    });
    test('x=[-1, -1, 3, 0],dout = [10,2,22,0] backward() should return [0, 0, 22, 0]', () => {
      const relu = new Relu(nj.array([0, 0, 1, 0]));
      expect(relu.backward(nj.array([10, 2, 22, 0])).tolist()).toEqual([
        0, 0, 22, 0,
      ]);
    });
  });
  describe('Relu.backwardBatch', () => {
    test('x=[[10,21,0,-1],[-1,-1,3,0]] dout=[[-1,10,10,0],[10,2,22,0]] backwardBatch() should return [[-1, 10, 0, 0],[0, 0, 22, 0]]', () => {
      const relu = new Relu(
        undefined,
        nj.array([
          [1, 1, 0, 0],
          [0, 0, 1, 0],
        ])
      );
      expect(
        relu
          .backwardBatch(
            nj.array([
              [-1, 10, 10, 0],
              [10, 2, 22, 0],
            ])
          )
          .tolist()
      ).toEqual([
        [-1, 10, 0, 0],
        [0, 0, 22, 0],
      ]);
    });
  });
});
