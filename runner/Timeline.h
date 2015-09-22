#pragma once

#include <string.h>
#include <vector>
#include <map>
#include <algorithm>
#include "proj_defs.h"

using namespace std;

class Timeline {
public:
    struct frame_t {
        float step;
        vector<float> temps;

        frame_t(float step, const vector<float>& temps);
        frame_t(const frame_t& other);
    };

    Timeline();
    void reset();
    void addFrame(frame_t& frameToAdd);
    float getCurTime() const;
    vector<float> getPoints() const;
    int findTempForPos(float pos, vector<float>& temps);

private:
    float curTime;
    float prevStep;
    map<const float, frame_t> frames;
};