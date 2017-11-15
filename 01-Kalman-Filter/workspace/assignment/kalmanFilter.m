function [ predictx, predicty, state, param ] = kalmanFilter( t, x, y, state, param, previous_t )
    %% State definition:
    %   [position_x, position_y, velocity_x, velocity_y]

    %% Estimation parameters:

    %% First measurement initialization:
    if previous_t<0
        % Initialize system process:
        state = [x, y, 0, 0];
        param.P = [
          0.01, 0.00,    0.00,    0.00;
          0.00, 0.01,    0.00,    0.00;
          0.00, 0.00,  100.00,    0.00;
          0.00, 0.00,    0.00,  100.00
        ];
        param.var_ax = 0.09;
        param.var_ay = 0.09;
        % Initialize measurement process:
        param.H = [
          1.0, 0.0, 0.0, 0.0;
          0.0, 1.0, 0.0, 0.0
        ];
        param.R = [
          0.0001, 0.0000;
          0.0000, 0.0001
        ];
        predictx = x;
        predicty = y;
        return;
    end

    %% Kalman filter estimation:
    % Time elapsed:
    delta_t = t - previous_t;

    % System matrix:
    F = [
      1.0, 0.0, delta_t,     0.0;
      0.0, 1.0,     0.0, delta_t;
      0.0, 0.0,     1.0,     0.0;
      0.0, 0.0,     0.0,     1.0
    ];
    % System noise:
    Q = [
      0.25*delta_t^4*param.var_ax,                         0.0, 0.50*delta_t^3*param.var_ax,                         0.0;
                              0.0, 0.25*delta_t^4*param.var_ay,                         0.0, 0.50*delta_t^3*param.var_ay;
      0.50*delta_t^3*param.var_ax,                         0.0,      delta_t^2*param.var_ax,                         0.0;
                              0.0, 0.50*delta_t^3*param.var_ay,                         0.0,      delta_t^2*param.var_ay
    ];

    % Predict:
    state = F * state';
    param.P = F * param.P * F' + Q;

    % Kalman gain:
    K = param.P * param.H' * pinv(param.H * param.P * param.H' + param.R);

    % Update:
    z = [x, y]';
    state = state + K * (z - param.H * state);
    param.P = (eye(4) - K * param.H) * param.P;

    % Set prediction:
    predictx = state(1) + 0.330*state(3);
    predicty = state(2) + 0.330*state(4);
    state = state';
end
