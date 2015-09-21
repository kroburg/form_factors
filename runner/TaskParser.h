// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include <string>
#include <iostream>
#include <vector>
#include <regex>
#include <assert.h>

using namespace std;

class TaskParser {
public:
    // @brief Callback on parsing end
    // @details 1st param is state of parsing (0 if OK), second is total number of frames
    typedef void(*onEndCb)(int, int);
    // @brief Callback on each frame end
    // @details 1st param is current frame number, second is total number of frames at current state, float is step value and vector is temperature data for all faces
    typedef void(*onEndFrameCb)(int, int, float, vector<float>&);

    TaskParser(onEndCb onEnd, onEndFrameCb onFrameEnd);

    // @brief call this on each incoming line
    int onLine(string line);

    // @brief resets all parser state
    void reset();

private:
    int curFrame;
    int totalFrames;
    regex newFrame_re;
    regex step_re;
    regex tmp_re;
    regex comment_re;
    vector<float> faceTemps;
    onEndCb onEnd;
    onEndFrameCb onFrameEnd;
    float curStep;

    enum state {
        FINISHED,
        RUNNING,
        ERROR,
        FRAME
    };
    state curState;
};