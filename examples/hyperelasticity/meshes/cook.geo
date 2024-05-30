// This code was created by pygmsh v6.0.2.
lc0 = 10.0;
lc1 = 0.5*lc0;
lc2 = 0.3*lc0;
lc3 = 0.15*lc0;

p0 = newp;
Point(p0) = {0.0, 0.0, 0.0, lc0};
p1 = newp;
Point(p1) = {48.0, 44.0, 0.0, lc1};
p2 = newp;
Point(p2) = {48.0, 60.0, 0.0, lc2};
p3 = newp;
Point(p3) = {0.0, 44.0, 0.0, lc3};
l0 = newl;
Line(l0) = {p0, p1};
l1 = newl;
Line(l1) = {p1, p2};
l2 = newl;
Line(l2) = {p2, p3};
l3 = newl;
Line(l3) = {p3, p0};
ll0 = newll;
Line Loop(ll0) = {l0, l1, l2, l3};
rs0 = news;
Surface(rs0) = {ll0};

Transfinite Curve {3, 1} = 59 Using Progression 1;
Transfinite Curve {4, 2} = 27 Using Progression 1;

Transfinite Surface {6} Alternated;

Physical Line(1) = {l0};
Physical Line(2) = {l2};
Physical Line(3) = {l1};
Physical Line(4) = {l3};
Physical Surface(0) = {rs0};


