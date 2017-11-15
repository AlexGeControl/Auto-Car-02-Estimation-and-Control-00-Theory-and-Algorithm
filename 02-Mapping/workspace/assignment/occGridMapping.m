% Robotics: Estimation and Learning
% WEEK 3
%
% Occupancy Grid Mapping.

function map = occGridMapping(ranges, scanAngles, pose, param)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parameters
    %
    % % the number of grids for 1 meter.
    % param.resol
    % % the initial map size in pixels
    % param.size
    % % the origin of the map in pixels
    % param.origin
    %
    % Log-odd parameters
    % param.lo_occ;
    % param.lo_free;
    % param.lo_max;
    % param.lo_min;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialize map
    map = zeros(param.size);

    M = size(ranges,1);
    N = size(pose,2);

    for i = 1:N
        % Measurement source:
        source = ceil( ...
            param.resol * [pose(1,i),pose(2,i)] ...
        ) + param.origin';

        % Find grids hit by the rays (in the gird map coordinate):
        occ = ceil( ...
            param.resol * [
                ranges(:,i).*cos(pose(3,i)+scanAngles)+pose(1,i), -ranges(:,i).*sin(pose(3,i)+scanAngles)+pose(2,i)
            ] ...
        ) + param.origin';

        % Update log-odds for occupied-measurement cells:
        map(occ(:,2),occ(:,1)) = map(occ(:,2),occ(:,1)) + param.lo_occ;

        % Update log-odds for free-measurement cells:
        for j = 1:M
            % Find free-measurement cells
            [freex, freey] = bresenham(source(1),source(2),occ(j,1),occ(j,2));
            % Update the log-odds
            free = sub2ind(size(map),freey,freex);
            map(free) = map(free) - param.lo_free;
        end

        % Saturate the log-odd values
        map = max(map,param.lo_min);
        map = min(map,param.lo_max);
    end
end
