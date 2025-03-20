function H = proj2Affine(Vp1, Vp2)
% function H = reTransf(Vp1, Vp2)
%
% This function defines a homography that maps the points that lie on the line
% defined by the homogeneous 2D points Vp1 and Vp2 to a line at the infinity
%
% Parameters:
%
% Vp1, Vp2: 3x1 matrices, that represents homogenous 2D points.
%
% Returns a 3x3 matrix that represents the mapping homography.
%

% First we compute the equation of the line where Vp1 and Vp2 lies.
	m = [Vp1(1), Vp1(2);Vp2(1), Vp2(2)];
	b = [-1; -1];
	l = ones(3,1);
	l(1:2) = m \ b;

% Define H according to formula in Hartley & Zisserman "Multiple View Geometry"
% 2nd edition, section 2.7.2, pp 49
	H=eye(3);
	H(3,:) = l';
end