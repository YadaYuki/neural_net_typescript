/*

*/
export interface Layer {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  forward(...arg: any): void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  backward(...arg: any): void;
}
