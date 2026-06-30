"""
Generated Phase 9 farmer-facing hardening cases.

These cases extend the checked-in Phase 4 JSONL set without adding unofficial
forum/social text to the trusted corpus. They are evaluation prompts only.
"""

from __future__ import annotations


def _case(
    question: str,
    language: str,
    topic: str,
    expected_route: str,
    expected_source_type: str,
    reference_answer: str,
    source_basis: str,
) -> dict[str, str]:
    return {
        "question": question,
        "language": language,
        "topic": topic,
        "expected_route": expected_route,
        "expected_source_type": expected_source_type,
        "reference_answer": reference_answer,
        "source_basis": source_basis,
    }


def generate_phase9_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []

    cases.extend(
        _case(q, lang, "pm_kisan", "rag", "official_pdf", "Use PM-KISAN official eligibility, amount, exclusion, Aadhaar, land-record, and registration guidance.", "PM-KISAN FAQ")
        for q, lang in [
            ("PM-KISAN eligibility landholding farmer family ke liye simple batao", "hinglish"),
            ("PM-KISAN mein 6000 rupees teen instalment ka rule kya hai?", "hinglish"),
            ("PM-KISAN registration mein Aadhaar aur bank details kyu chahiye?", "hinglish"),
            ("PM-KISAN exclusion mein government employee farmer cover hota hai kya?", "hinglish"),
            ("PM-KISAN land record mismatch ho to kya documents verify hote hain?", "hinglish"),
            ("PM-KISAN family definition husband wife minor children ke hisab se kya hai?", "hinglish"),
            ("PM-KISAN self registration ke liye farmer ko kaunsi basic details chahiye?", "hinglish"),
            ("PM-KISAN eKYC pending hai to official FAQ kya bolta hai?", "hinglish"),
            ("Explain PM-KISAN eligibility for a small landholding farmer family.", "english"),
            ("What is the PM-KISAN annual amount and how many instalments are paid?", "english"),
            ("Which documents are normally needed for PM-KISAN verification?", "english"),
            ("Does PM-KISAN exclude income tax paying farmer families?", "english"),
            ("What does the PM-KISAN FAQ say about Aadhaar-based verification?", "english"),
            ("Can institutional landholders receive PM-KISAN benefits?", "english"),
            ("Where should PM-KISAN beneficiary records come from?", "english"),
            ("PM-KISAN yojana chhota kisan family lai eligibility ki hai?", "regional_romanized"),
        ]
    )
    cases.extend(
        _case(q, lang, "pm_kisan", "dynamic_router", "live_portal", "Farmer-specific PM-KISAN status and payment checks must route to the official live portal.", "dynamic PM-KISAN route")
        for q, lang in [
            ("Mera PM-KISAN registration status abhi check karo", "hinglish"),
            ("PM-KISAN meri kist account mein credit hui ya nahi?", "hinglish"),
            ("Check my PM-KISAN beneficiary list status now.", "english"),
            ("PM-KISAN kist kab credit hogi mere bank account mein?", "regional_romanized"),
        ]
    )

    cases.extend(
        _case(q, lang, "pmfby", "rag", "official_pdf", "Use PMFBY official guidelines for premium, enrolment, intimation, documents, localised loss, and claim settlement.", "PMFBY Operational Guidelines")
        for q, lang in [
            ("PMFBY flood damage claim 72 hours mein inform karna hota hai kya?", "hinglish"),
            ("Fasal bima kharif crop premium 2 percent ka rule samjhao", "hinglish"),
            ("PMFBY prevented sowing loss ka claim kab apply hota hai?", "hinglish"),
            ("PMFBY localized calamity mein farmer ko kisko intimation deni hai?", "hinglish"),
            ("Fasal bima claim ke liye crop cutting experiment ka role kya hai?", "hinglish"),
            ("PMFBY drought loss mein claim settlement ka official process kya hai?", "hinglish"),
            ("PMFBY non loanee farmer enrolment ke liye kya documents chahiye?", "hinglish"),
            ("PMFBY post harvest loss ke liye farmer kya kare?", "hinglish"),
            ("What is the farmer premium for kharif crops under PMFBY?", "english"),
            ("How should a farmer report localised flood loss under PMFBY?", "english"),
            ("What documents support a PMFBY crop insurance claim?", "english"),
            ("When does PMFBY cover prevented sowing or failed sowing?", "english"),
            ("How are PMFBY claims linked to crop yield estimation?", "english"),
            ("What exclusions should a farmer know under PMFBY?", "english"),
            ("How does PMFBY handle post-harvest crop loss?", "english"),
            ("Fasal bima flood nuksan lai farmer nu kinne time vich batana hai?", "regional_romanized"),
            ("PMFBY claim document kis kis cheez ka proof mangta hai?", "regional_romanized"),
            ("Kharif fasal bima premium farmer share kitna hota hai?", "regional_romanized"),
            ("Sukha padne par PMFBY claim process kaise chalega?", "regional_romanized"),
            ("PMFBY local calamity mein insurance company ko kaise inform kare?", "regional_romanized"),
        ]
    )

    cases.extend(
        _case(q, lang, "kcc_credit", "rag", "vikaspedia", "Use Kisan Credit Card/Vikaspedia guidance for application, credit limit, purpose, repayment, interest, and collateral.", "Vikaspedia KCC")
        for q, lang in [
            ("Kisan Credit Card crop loan ke liye apply kaise karna hai?", "hinglish"),
            ("KCC mein collateral free loan limit kya hoti hai?", "hinglish"),
            ("KCC interest subvention ka benefit farmer ko kaise milta hai?", "hinglish"),
            ("Kisan Credit Card se dairy aur fisheries purpose cover hote hain kya?", "hinglish"),
            ("KCC credit limit cropping pattern ke hisab se kaise decide hoti hai?", "hinglish"),
            ("KCC renewal aur validity ke bare mein simple batao", "hinglish"),
            ("KCC repayment time crop season se linked hota hai kya?", "hinglish"),
            ("How can a farmer apply for a Kisan Credit Card?", "english"),
            ("What purposes can KCC credit be used for?", "english"),
            ("What is collateral-free lending under KCC?", "english"),
            ("How is the KCC credit limit assessed?", "english"),
            ("What interest assistance is available for crop loans through KCC?", "english"),
            ("How does repayment work for a Kisan Credit Card crop loan?", "english"),
            ("Can allied agriculture activities be covered by KCC?", "english"),
            ("KCC loan limit kheti area ke hisab se banti hai kya?", "regional_romanized"),
            ("Kisan credit card vich collateral free madad kinni ho sakdi hai?", "regional_romanized"),
            ("KCC interest subsidy time par repayment se judi hai kya?", "regional_romanized"),
            ("KCC banwane ke liye bank ko kya form dena hota hai?", "regional_romanized"),
        ]
    )

    cases.extend(
        _case(q, lang, "land_fra_legal", "rag", "official_pdf", "Use FRA/LARR official sources for forest rights, Gram Sabha process, land acquisition compensation, consent, and rehabilitation.", "FRA/LARR official documents")
        for q, lang in [
            ("Forest Rights Act mein individual forest right claim ka process kya hai?", "hinglish"),
            ("Gram Sabha FRA claim verify karne mein kya role nibhati hai?", "hinglish"),
            ("Community forest resource rights ka matlab kya hota hai?", "hinglish"),
            ("Land acquisition mein affected family ko compensation kaise milta hai?", "hinglish"),
            ("LARR Act rehabilitation and resettlement entitlement kya hota hai?", "hinglish"),
            ("Zameen adhigrahan mein consent aur social impact assessment kab chahiye?", "hinglish"),
            ("FRA title milne ke baad forest land par farmer ka kya adhikar hota hai?", "hinglish"),
            ("What is the Gram Sabha role under the Forest Rights Act?", "english"),
            ("What rights can forest dwelling Scheduled Tribes claim under FRA?", "english"),
            ("How does LARR handle compensation for land acquisition?", "english"),
            ("What rehabilitation benefits are mentioned for displaced families?", "english"),
            ("When is social impact assessment relevant in land acquisition?", "english"),
            ("What are community forest resource rights?", "english"),
            ("How should a farmer understand consent under land acquisition law?", "english"),
            ("Jangal adhikar claim mein Gram Sabha kya karti hai?", "regional_romanized"),
            ("Zameen acquisition par parivar ko rehabilitation mil sakta hai kya?", "regional_romanized"),
            ("FRA community rights gaon ke liye kaise kaam karte hain?", "regional_romanized"),
            ("LARR compensation farmer family ko kis basis par milta hai?", "regional_romanized"),
        ]
    )

    cases.extend(
        _case(q, lang, "state_schemes", "rag", "state_scheme", "Use indexed state-scheme sources and prefer the named state over generic central scheme chunks.", "state scheme source")
        for q, lang in [
            ("Maharashtra Namo Shetkari PM-KISAN ke upar extra paisa deta hai kya?", "hinglish"),
            ("Punjab agriculture scheme mein farmer health ya insurance support kya hai?", "hinglish"),
            ("Bihar state farmer scheme mein diesel subsidy type madad kaise milti hai?", "hinglish"),
            ("Rajasthan farmer ko state agriculture scheme se kaunsi support milti hai?", "hinglish"),
            ("Telangana Rythu Bandhu type state support farmer ke liye kya hai?", "hinglish"),
            ("Gujarat kisan state scheme mein irrigation ya subsidy support kya hai?", "hinglish"),
            ("West Bengal Krishak Bandhu mein farmer ko kya benefit milta hai?", "hinglish"),
            ("Madhya Pradesh agriculture scheme farmer ke liye kaunsi madad batati hai?", "hinglish"),
            ("What extra income support does Maharashtra Namo Shetkari provide?", "english"),
            ("Which Punjab agriculture support source should answer a Punjab farmer question?", "english"),
            ("What state scheme support is indexed for Bihar farmers?", "english"),
            ("How should Rajasthan farmer scheme questions avoid PM-KISAN as top answer?", "english"),
            ("What Telangana farmer scheme support appears in the indexed state sources?", "english"),
            ("What Gujarat agriculture scheme support is available in the state source?", "english"),
            ("What benefit does West Bengal Krishak Bandhu provide to farmers?", "english"),
            ("Which Madhya Pradesh agriculture scheme source should be preferred?", "english"),
            ("Maharashtra shetkari ko Namo Shetkari se extra labh milta hai kya?", "regional_romanized"),
            ("Punjab kisan lai state scheme vich ki madad hai?", "regional_romanized"),
            ("Bihar kisan diesel subsidy jaisi state madad ke bare mein batao", "regional_romanized"),
            ("Rajasthan kisan yojana mein rajya source se jawab do", "regional_romanized"),
            ("Telangana Rythu Bandhu farmer support kaise milta hai?", "regional_romanized"),
            ("West Bengal Krishak Bandhu kisan benefit kya hai?", "regional_romanized"),
        ]
    )

    cases.extend(
        _case(q, lang, "mandi_weather_live", "dynamic_router", "live_portal", "Route live price, weather, rain, and spraying decisions to allowlisted live tools or official live portal fallback.", "dynamic live-data route")
        for q, lang in [
            ("Aaj soyabean ka mandi bhav Maharashtra mein kya hai?", "hinglish"),
            ("Aaj wheat ka mandi rate Punjab mein batao", "hinglish"),
            ("Kal Jaipur mein baarish hogi kya spraying karu?", "hinglish"),
            ("Pune weather forecast dekh kar batao spray safe hai kya?", "hinglish"),
            ("Aaj onion mandi price Nashik side kya chal raha hai?", "hinglish"),
            ("Cotton ka aaj ka mandi bhav Gujarat mein check karo", "hinglish"),
            ("Delhi weather tomorrow rain chance kya hai?", "hinglish"),
            ("Aaj potato mandi rate West Bengal mein kya hai?", "hinglish"),
            ("Rice mandi price Bihar mein live check karo", "hinglish"),
            ("Kal Nagpur mein rain aur wind spray ke liye safe hai kya?", "hinglish"),
            ("Mera crop spray karna hai, Pune ka mausam aaj kya hai?", "hinglish"),
            ("Aaj dhan ka mandi bhav Rajasthan mein batao", "hinglish"),
            ("What is today's soybean mandi price in Maharashtra?", "english"),
            ("Check live wheat market rate in Punjab today.", "english"),
            ("Will it rain in Jaipur tomorrow, and is spraying safe?", "english"),
            ("What is the live onion mandi price near Nashik today?", "english"),
            ("Check cotton mandi rate in Gujarat today.", "english"),
            ("What is Delhi rain chance tomorrow for spraying?", "english"),
            ("Give live paddy market rate for Bihar today.", "english"),
            ("Is spraying safe in Pune based on today's weather?", "english"),
            ("Soyabean mandi bhav aaj Maharashtra side kitna hai?", "regional_romanized"),
            ("Punjab wheat market rate ajj live batao", "regional_romanized"),
            ("Jaipur kal baarish chance hai kya spray ke liye?", "regional_romanized"),
            ("Nashik onion bhav aaj mandi mein kya chal raha hai?", "regional_romanized"),
            ("Gujarat cotton ka live market rate check karo", "regional_romanized"),
            ("Pune mausam dekh ke spraying safe hai kya?", "regional_romanized"),
        ]
    )

    cases.extend(
        _case(q, lang, "crop_soil_advisory", "rag", "official_portal", "Use indexed official advisory/portal sources for soil health, irrigation, micro-irrigation, farm labour, market linkage, and crop guidance.", "official agriculture portal")
        for q, lang in [
            ("Soil Health Card se farmer ko fertilizer advice kaise milti hai?", "hinglish"),
            ("PMKSY micro irrigation subsidy ka basic purpose kya hai?", "hinglish"),
            ("Drip irrigation scheme farmer water saving mein kaise help karti hai?", "hinglish"),
            ("Agriculture Infrastructure Fund se farmer group ko kya support mil sakta hai?", "hinglish"),
            ("eNAM market linkage farmer ko mandi selling mein kaise help karta hai?", "hinglish"),
            ("MNREGA farm pond ya labour support agriculture mein kaise useful hai?", "hinglish"),
            ("Cotton pest attack par static advisory source kya general guidance de sakta hai?", "hinglish"),
            ("Soil testing report ke basis par nutrient management kaise decide kare?", "hinglish"),
            ("Irrigation scheduling ke liye official crop advisory kaise use kare?", "hinglish"),
            ("Farm mechanization support farmer productivity mein kaise help kar sakta hai?", "hinglish"),
            ("How does the Soil Health Card help with fertiliser planning?", "english"),
            ("What is the purpose of micro-irrigation support under PMKSY?", "english"),
            ("How can drip irrigation reduce water use for farmers?", "english"),
            ("What does the Agriculture Infrastructure Fund support?", "english"),
            ("How can eNAM help farmers with market linkage?", "english"),
            ("What kind of farm labour support can official rural employment schemes provide?", "english"),
            ("What should a farmer do when pest attack guidance is not live-location specific?", "english"),
            ("How should soil test recommendations guide nutrient use?", "english"),
            ("What official source should answer irrigation advisory questions?", "english"),
            ("How can farm mechanization schemes support small farmers?", "english"),
            ("Why should pesticide spraying guidance be verified with local advisory?", "english"),
            ("Soil health card se khaad ka istemal kaise plan kare?", "regional_romanized"),
            ("PMKSY drip irrigation paani bachane mein madad karti hai kya?", "regional_romanized"),
            ("eNAM market linkage farmer sale ke liye kaise kaam karta hai?", "regional_romanized"),
            ("AIF se storage ya infrastructure support kaise mil sakta hai?", "regional_romanized"),
            ("Cotton pest advice local krishi kendra se verify karna chahiye kya?", "regional_romanized"),
        ]
    )

    return cases
