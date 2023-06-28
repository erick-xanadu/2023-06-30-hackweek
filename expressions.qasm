
gate h q {
  U(1.57079, 0.0, 3.141592) q;
  gphase(-0.78539);
}

gate X q1 {
  U(3.14159, 0.0, 3.14159) q1;
  gphase(-1.57079);
}

gate rx(theta) q12 {
  U(theta, -1.57079, 1.57079) q12;
  gphase(-theta / 2.0);
}

gate rz(lambda) q123 {
  gphase(-lambda / 2.0);
  U(0.0, 0.0, lambda) q123;
}

gate crz(theta) q1, q2 {
  ctrl @ rz(theta) q1, q2;
}

qubit x;
rx(3.14159) x;
