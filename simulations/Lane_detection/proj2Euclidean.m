function [l, lp, p, pp, Hl] = proj2Euclidean(Vp1, Vp2, Cy)
%
% function [l, lp, p, pp, Hl] = proj2Euclidean(Vp1, Vp2, Cy)
%
% This function computes the homography that maps the points on an
% image where projective features are present, e.g. there are vanishing points
% to points on an image that corresponds to an aerial vie over the plane where
% those to vanishing points exist.
%
% Parameters:
%
% Vp1, Vp2: 3x1 matrices, that represents homogenous 2D coordinates of the
%           vanishing points. It is assumed that the vanishing points
%           lie on the horizon, which in this case is defined as an horizontal
%           line that is at a Cy distance from the origin.
%
% Return:
%
% Extra: Parámetros de prueba: Cy, vp1 y vp2
%
% Cy = 540
% vp1 = [1916, 540, 1]
% vp2 = [  -3, 540, 1]
%

% Compute the transformation that maps the projective space into and
% affine one.
	H = proj2Affine(Vp1, Vp2);

% Define 4 random points on a line that is parallel to the horizon.
	xc = round((Vp1(1)+Vp2(1))/2)
	p = ones(3,4);
	p(1, :) = xc + [-600:400:600];
	p(2, :) = round(Cy+Cy/2);


% The lines that passes through the points p and the vanishing points are
% computed and normalized.
	l = ones(3,4);
	l(:, 1) = cross(Vp1, p(:,2));
	l(:, 2) = cross(Vp1, p(:,1));
	l(:, 3) = cross(Vp2, p(:,3));
	l(:, 4) = cross(Vp2, p(:,4));
	l = normPoints(l);

% The points that intersect the lines in the projective space are computed
% and normalized.
	p(:,1) = cross(l(:,2), l(:,3));
	p(:,2) = cross(l(:,1), l(:,3));
	p(:,3) = cross(l(:,1), l(:,4));
	p(:,4) = cross(l(:,2), l(:,4));
    p = normPoints(p);

% The mapping that maps lines is defined in terms of the mapping of points.
	Hl = inv(H)';
	
% The computed lines are mapped into the affine space.
	lp = Hl * l;

% The points that intersect the lines in the affine space are computed
% and normalized.
	pp=ones(3,4);
	pp(:,1) = cross(lp(:,2), lp(:,3));
	pp(:,2) = cross(lp(:,1), lp(:,3));
	pp(:,3) = cross(lp(:,1), lp(:,4));
	pp(:,4) = cross(lp(:,2), lp(:,4));
    pp = normPoints(pp);


 % Esta parte de aqui Ya no debe estar. Debemos calcular la homografia entre los cuatro puntos
 % pp y los cuatro puntos que definen la reticula. Esto es mas facil hacerlo en openCV
 % utilizando la función findHomography
 %
 % https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
 %

 % An homography that translate the points in the affine space in such
 % way that minimizes de difference to the points in the projective space
 % is computed.
 %
 % This should be substituted so that all the parameters of the homography
 % is computed in a single operation.
    pm = mean(p, 2);
    ppm = mean(pp, 2);
    dfp = pm - ppm;
    HA=eye(3);
    HA(1:2, 3) = dfp(1:2);
    pp = HA * pp;


% The points and lines found are drawn.
	figure(1);
	clf();
	plot(Vp1(1), Vp1(2), "*r", Vp2(1), Vp2(2), "*r");
	hold on
	for idx=1:4
		plot(p(1,idx), p(2,idx), "*k");
		plot(pp(1,idx), pp(2,idx), "*g");
	end

	xmin = min([Vp1(1), Vp2(1)]);
	xmax = max([Vp1(1), Vp2(1)]);
	xrange = xmax-xmin;
	inc = xrange /(100-1);
	x=xmin:inc:xmax;
	for idx=1:4	
		m = -l(1, idx) / l(2, idx);
		b = -l(3, idx) / l(2, idx);
		y = m * x + b;
		plot(x,y,'b');
	end	

	axis ij
	axis equal
end