

Ancho = 5;
midAncho = Ancho / 2;
Alto = 8;
midAlto = Alto / 2;
SQ = [-midAncho, midAncho, midAncho,-midAncho, -midAncho;
              0,        0,        0,        0,         0;
	   -midAlto, -midAlto,  midAlto,  midAlto,  -midAlto;
              1,        1,        1,        1,         1];


K = [1000,    0, 960;
        0, -1000, 540;
        0,    0,   1];

T = [0, -2, 10]';
theta = pi / 16;
theta = 0;
cs=cos(theta);
sn=sin(theta);
Ry = [cs, 0, sn;
      0, 1,  0;
    -sn, 0, cs];
Rx = [0,  0,   0;
      1, cs, -sn;
      0, sn,  cs];
Rz = [cs, -sn, 0;
      sn,  cs, 0;
       0,   0, 1];
R0=eye(3);

for theta =0:2:720       	
	G=[R0, T; 0,0,0,1];
	th = theta * pi / 180; 
	cs=cos(th);
	sn=sin(th);
	Rg = [cs, 0, sn;
           0, 1,  0;
         -sn, 0, cs];
    Gg=[Rg,zeros(3,1);0,0,0,1];
    SqG = Gg * SQ;
	P = K * [eye(3),zeros(3,1)] * G * SqG;
	for i=1:5
		P(:,i) /= P(3,i);
	end
	clf
	hold on;
	L1 = cross (P(:,1),P(:,2));
	L2 = cross (P(:,2),P(:,3));
	L3 = cross (P(:,3),P(:,4));
	L4 = cross (P(:,4),P(:,5));
	VP1 = cross(L1,L3);
	VP2 = cross(L2,L4);
	if abs(VP1(3)) > 1e-2
		VP1 = VP1/VP1(3);
		line ([P(1,1), VP1(1)], [P(2,1), VP1(2)]);
    	line ([P(1,4), VP1(1)], [P(2,4), VP1(2)]);
    	plot(VP1(1), VP1(2), "m*");
    end
    if abs(VP2(3)) > 1e-2
		VP2 = VP2/VP2(3);
		line ([P(1,3), VP2(1)], [P(2,3), VP2(2)]);
    	line ([P(1,4), VP2(1)], [P(2,4), VP2(2)]);
		plot(VP2(1), VP2(2), "m*");
	end
	plot(P(1,:), P(2,:), 'r', P(1,:), P(2,:), "or");
	axis ([0, 1920, 0, 1080], "square");
	axis ij;
	hold off
	pause(0.0033);
end