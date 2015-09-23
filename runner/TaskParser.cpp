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

#include "TaskParser.h"

TaskParser::TaskParser(onEndCb onEnd, onEndFrameCb onFrameEnd)
    : newFrame_re("^\\s{0,}newfrm\\s{1,}frame_(\\d+)$", regex_constants::icase),
      step_re("^\\s{0,}step\\s{1,}([\\d\\.]+)$"),
      tmp_re("^\\s{0,}tmprt\\s{1,}([\\d\\.]+)$", regex_constants::icase),
      comment_re("^\\s{0,}#.{0,}$", regex_constants::icase),
      curState(RUNNING) {
    assert(onEnd);
    assert(onFrameEnd);
    this->onEnd = onEnd;
    this->onFrameEnd = onFrameEnd;
    reset();
}

void TaskParser::reset() {
    curFrame = 0;
    curState = RUNNING;
    curStep = 0;
    totalFrames = 0;
}

int TaskParser::onLine(string line) {
    // Check for comments first
    if (curState != ERROR && !line.empty() && regex_match(line, comment_re)) {
        return 0;
    }
    smatch match;
    switch (curState) {
    case (ERROR) :
        return -1;
    case (FINISHED) :
        return 0;
    case(RUNNING):
        if (line.empty()) {
            return 0;
        } else {
            if (regex_search(line, match, newFrame_re) && match.size() > 1) {
                curFrame = stoi(match.str(1), NULL);
                curState = FRAME;
                curStep = 0.0f;
                faceTemps.clear();
                return 0;
            } else {
                return 0;
            }
        }
    case (FRAME):
        if (line.empty()) {
            curState = RUNNING;
            ++totalFrames;
            onFrameEnd(curFrame, totalFrames, curStep, faceTemps);
            return 0;
        }
        else if (regex_search(line, match, tmp_re) && match.size() > 1) {
            faceTemps.push_back(stof(match.str(1), NULL));
            return 0;
        }
        else if (regex_search(line, match, step_re) && match.size() > 1) {
            curStep = stof(match.str(1), NULL);
            return 0;
        }
    default:
        return -1;
    }
    return -1;
}