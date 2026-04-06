#pragma once

#include <string>
#include <format>
#include <print>
#include <source_location>
#include <type_traits>

namespace utils {
    template <typename... Args>
        struct LocFormatString {
        std::format_string<Args...> fmt;
        std::source_location loc;

        template <typename String>
        consteval LocFormatString(const String& s, std::source_location loc = std::source_location::current()) : fmt(s), loc(loc) {}
    };

    template <typename... Args>
    [[noreturn]] void error(LocFormatString<std::type_identity_t<Args>...> fmt_with_loc, Args&&... args) {
        const char* func_name = fmt_with_loc.loc.function_name();
        std::print("[\033[31mError\033[0m][{}]: ", func_name);
        std::println(fmt_with_loc.fmt, std::forward<Args>(args)...);

        std::abort();
    }

    template <typename... Args>
    void warn(LocFormatString<std::type_identity_t<Args>...> fmt_with_loc, Args&&... args) {
        const char* func_name = fmt_with_loc.loc.function_name();
        std::print("[\033[33mWarning\033[0m][{}]: ", func_name);
        std::println(fmt_with_loc.fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log(LocFormatString<std::type_identity_t<Args>...> fmt_with_loc, Args&&... args) {
        const char* func_name = fmt_with_loc.loc.function_name();
        std::print("[\033[32mLog\033[0m][{}]: ", func_name);
        std::println(fmt_with_loc.fmt, std::forward<Args>(args)...);
    }
}

namespace timing {
    struct TimingMetrics {
        int prefillTokens = 0;
        int decodeTokens = 0;

        double prefillTime = 0.0f;
        double decodeTime = 0.0f;

        [[nodiscard]] double prefillTPS() const {return prefillTokens/prefillTime;}
        [[nodiscard]] double decodeTPS() const {return decodeTokens/decodeTime;}

        void print() const {
            std::print("\nPrefill Tokens: {} in {}ms ({:.5f} tok/s)\n", prefillTokens, prefillTime, prefillTPS());
            std::print("Decode Tokens: {} in {}ms ({:.5f} tok/s)\n", decodeTokens, decodeTime, decodeTPS());
        }
    };

}