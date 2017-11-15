% Robotics: Estimation and Learning
% WEEK 4
%
% Complete this function following the instruction.
function pose = particleLocalization(ranges, scanAngles, map, param)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Pose Output
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Number of poses to calculate
    N = size(ranges, 2);
    % Output buffer
    pose = zeros(3, N);
    % Use the given initial pose into pose for j=1, ignoring the j=1 ranges.
    pose(:,1) = param.init_pose;
    % The pose(:,j) should be the pose when ranges(:,j) were measured.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Map Parameters
    %
    % % the number of grids for 1 meter.
    % param.resol
    % % the origin of the map in pixels
    % param.origin
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Particle Filter Params
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Process noise:
    motion_mean = zeros(1, 3)';
    motion_std = [0.0164,0.0178,0.0091]';

    % Number of particles
    M = 1000;

    % Create particles:
    P = repmat(pose(:,1), [1, M]);

    % Create particle weights:
    corr = zeros(1, M);
    w = 1.0 / M * ones(1, M);

    % Map setup:
    Y = size(map,1); X = size(map,2);

    % Lidar measurement downsample:
    D = 2;
    sample_ranges = ranges(1:D:end, :);
    sample_scan_angles = scanAngles(1:D:end);

    % Min. effective sample number:
    min_num_effective = ceil(0.618 * M);

    % Start estimating pose from the second measurement
    for i = 2:N
        % 1) Propagate the particles
        P = P + (motion_std .* randn(3, M)) + motion_mean;

        % 2) Measurement Update
        for j = 1:M
            %   2-0) Potential pose:
            potential_pose = P(:, j);
            %   2-1) Find measurement source:
            src = ceil(param.resol * [
                potential_pose(1), potential_pose(2)
            ]) + param.origin';
            %   2-2) Find grid cells hit by the rays (in the grid map coordinate frame):
            occ = ceil(param.resol * [
                +sample_ranges(:,i).*cos(potential_pose(3) + sample_scan_angles) + potential_pose(1), -sample_ranges(:,i).*sin(potential_pose(3) + sample_scan_angles) + potential_pose(2)
            ]) + param.origin';
            %   Bound occupancy grid coordinates:
            occ = max(occ, 1);
            occ(:,1) = min(occ(:,1), X); occ(:,2) = min(occ(:,2), Y);
            occ_pred = map(sub2ind(size(map),occ(:,2),occ(:,1)));
            %   Correlation score, occupancy grid part:
            corr(j) = 10 * sum(occ_pred > 0.5) - 5 * sum(occ_pred < 0.0);
            %   2-3) Find free cells between source and occupancy:
            for k = 1:size(occ,2)
                [freex, freey] = bresenham(src(1),src(2),occ(k,1),occ(k,2));
                free_pred = map(sub2ind(size(map),freey,freex));
            %   Correlation score, free grid part:
                corr(j) = corr(j) + 1 * sum(free_pred < 0.0) - 5 * sum(free_pred > 0.5);
            end
        end
        %   2-4) Update the particle weights
        w = M * corr .* w;
        %   2-5) Choose the best particle to update the pose
        w = w - max(w);
        w = exp(w) / sum(exp(w));
        [val_max, idx_max] = max(w);
        pose(:,i) = P(:,idx_max);

        % 3) Resample if the effective number of particles is smaller than a threshold
        num_effective = sum(w)^2 / sum(w.^2);
        disp("[Time]: " + i + " Num Effective--" + num_effective + " ML--" + val_max)
        if num_effective < min_num_effective
            disp("[Time]: " + i + " Resample")
            % Cumulative probability:
            cumw = cumsum(w);
            % Create resample indices:
            resample_idx = zeros(1, M);
            % Resample:
            for k = 1:M
                resample_idx(k) = find(cumw >= rand(), 1);
            end
            % Update:
            P = P(:, resample_idx);
            w = w(resample_idx);
            w = w / sum(w);
        end
        % 4) Visualize the pose on the map as needed
    end

end
