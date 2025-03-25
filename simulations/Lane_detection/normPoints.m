function p = normPoints(p)
	[r,c] = size(p);
	for idx = 1:c
		p(:,idx) = p(:,idx)/p(r,idx);
	end
end