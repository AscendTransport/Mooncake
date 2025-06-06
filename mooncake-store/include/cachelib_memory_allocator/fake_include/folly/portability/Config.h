/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if __has_include(<features.h>)
#include <features.h> // @manual
#endif

#if __has_include(<bits/c++config.h>)
#include <bits/c++config.h> // @manual
#endif

#if __has_include(<__config>)
#include <__config> // @manual
#endif

#ifdef __ANDROID__
#include <android/api-level.h> // @manual
#endif

#ifdef __APPLE__
#include <Availability.h> // @manual
#include <AvailabilityMacros.h> // @manual
#include <TargetConditionals.h> // @manual
#endif
