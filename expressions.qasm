
gate h q {
  float pi = 3.141592653589793115997963468544185161590576171875;
  U(pi/2.0, 0.0, pi) q;
  gphase(-pi/4.0);
}

gate X q {
  float pi = 3.141592653589793115997963468544185161590576171875;
  U(pi, 0.0, pi) q;
  gphase(-pi/2.0);
}

gate rx(theta) q {
  float pi = 3.141592653589793115997963468544185161590576171875;
  U(theta, -pi/2.0, pi/2.0) q;
  gphase(-theta / 2.0);
}

gate rz(lambda) q {
  gphase(-lambda / 2.0);
  U(0.0, 0.0, lambda) q;
}

gate crz(theta) q1, q2 {
  ctrl @ rz(theta) q1, q2;
}

float pi = 3.141592653589793115997963468544185161590576171875;
qubit x;
qubit y;
h x;
h y;
