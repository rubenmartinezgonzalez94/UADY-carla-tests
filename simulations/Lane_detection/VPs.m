

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

T0 = [0, -2, 20]';
TX = [-10, 0, 0;0, 0, 0;10, 0 ,0]' 
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
	Ga=[R0, T0+TX(:, 1); 0,0,0,1];
	Gb=[R0, T0+TX(:, 2); 0,0,0,1];
	Gc=[R0, T0+TX(:, 3); 0,0,0,1];
	th = theta * pi / 180; 
	cs=cos(th);
	sn=sin(th);
	Rg = [cs, 0, sn;
           0, 1,  0;
         -sn, 0, cs];
    Gg=[Rg,zeros(3,1);0,0,0,1];
    SqG = Gg * SQ;
	Pa = K * [eye(3),zeros(3,1)] * Ga * SqG;
	Pb = K * [eye(3),zeros(3,1)] * Gb * SqG;
	Pc = K * [eye(3),zeros(3,1)] * Gc * SqG;
	for i=1:5
		Pa(:,i) /= Pa(3,i);
		Pb(:,i) /= Pb(3,i);
		Pc(:,i) /= Pc(3,i);
	end
	clf
	hold on;

	%Procesando el cuadrado A
	L1a = cross (Pa(:,1),Pa(:,2));
	L2a = cross (Pa(:,2),Pa(:,3));
	L3a = cross (Pa(:,3),Pa(:,4));
	L4a= cross (Pa(:,4),Pa(:,5));
	VP1a = cross(L1a,L3a);
	VP2a = cross(L2a,L4a);
	if abs(VP1a(3)) > 1e-2
		VP1a = VP1a/VP1a(3);
		line ([Pa(1,1), VP1a(1)], [Pa(2,1), VP1a(2)]);
    	line ([Pa(1,4), VP1a(1)], [Pa(2,4), VP1a(2)]);
    	plot(VP1a(1), VP1a(2), "m*");
    end
    if abs(VP2a(3)) > 1e-2
		VP2a = VP2a/VP2a(3);
		line ([Pa(1,3), VP2a(1)], [Pa(2,3), VP2a(2)]);
    	line ([Pa(1,4), VP2a(1)], [Pa(2,4), VP2a(2)]);
		plot(VP2a(1), VP2a(2), "m*");
	end
	plot(Pa(1,:), Pa(2,:), 'r', Pa(1,:), Pa(2,:), "or");

	%Procesando el cuadrado B
	L1b = cross (Pb(:,1),Pb(:,2));
	L2b = cross (Pb(:,2),Pb(:,3));
	L3b = cross (Pb(:,3),Pb(:,4));
	L4b= cross (Pb(:,4),Pb(:,5));
	VP1b = cross(L1b, L3b);
	VP2b = cross(L2b, L4b);
	if abs(VP1b(3)) > 1e-2
		VP1b = VP1b/VP1b(3);
		line ([Pb(1,1), VP1b(1)], [Pb(2,1), VP1b(2)]);
    	line ([Pb(1,4), VP1b(1)], [Pb(2,4), VP1b(2)]);
    	plot(VP1b(1), VP1b(2), "m*");
    end
    if abs(VP2b(3)) > 1e-2
		VP2b = VP2b/VP2b(3);
		line ([Pb(1,3), VP2b(1)], [Pb(2,3), VP2b(2)]);
    	line ([Pb(1,4), VP2b(1)], [Pb(2,4), VP2b(2)]);
		plot(VP2b(1), VP2b(2), "m*");
	end
	plot(Pb(1,:), Pb(2,:), 'r', Pb(1,:), Pb(2,:), "or");

   %Procesando el cuadrado C
	L1c = cross (Pc(:,1),Pc(:,2));
	L2c = cross (Pc(:,2),Pc(:,3));
	L3c = cross (Pc(:,3),Pc(:,4));
	L4c= cross (Pc(:,4),Pc(:,5));
	VP1c = cross(L1c, L3c);
	VP2c = cross(L2c, L4c);
	if abs(VP1c(3)) > 1e-2
		VP1c = VP1c/VP1c(3);
		line ([Pc(1,1), VP1c(1)], [Pc(2,1), VP1c(2)]);
    	line ([Pc(1,4), VP1c(1)], [Pc(2,4), VP1c(2)]);
    	plot(VP1c(1), VP1c(2), "m*");
    end
    if abs(VP2c(3)) > 1e-2
		VP2c = VP2c/VP2c(3);
		line ([Pc(1,3), VP2c(1)], [Pc(2,3), VP2c(2)]);
    	line ([Pc(1,4), VP2c(1)], [Pc(2,4), VP2c(2)]);
		plot(VP2c(1), VP2c(2), "m*");
	end
	plot(Pc(1,:), Pc(2,:), 'r', Pc(1,:), Pc(2,:), "or");

	axis ([0, 1920, 0, 1080], "square");
	axis ij;
	hold off
	pause(0.0033);
end
